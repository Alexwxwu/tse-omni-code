import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import cv2 as cv
import random
import soundfile as sf
import librosa
import sys
from tools import audioread
import scipy.io.wavfile as wavfile
from mel_spectrogram import MelSpec
import pdb

# External visual-frontend dependency (SEMO-621 repo): provides VisualFrontend
# and Visual_perturb helpers used by the switch/occlusion datasets.
sys.path.append('/mnt/users/hccl.local/wwu/SEMO-621/')
from visual_frontend.visual_frontend import VisualFrontend
sys.path.append('/mnt/users/hccl.local/wwu/SEMO-621/data_preparation/')
from Visual_perturb import *

EPS = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max
np.random.seed(0)
random.seed(0)


class dataset(data.Dataset):
    def __init__(
        self,
        vsr_feat_dir,
        vsr_sync_lst_train_path,
        vsr_sync_lst_path,
        visual_direc,
        aux_list,
        aux_train_list,
        mix_lst_path,
        mix_lst_train_path,
        codec_lst_path, 
        codec_lst_train_path, 
        visual_codec_lst_train_path,
        batch_size,
        partition="val",
        sampling_rate=16000,
        codec_rate = 25,
        mix_no=2,
        max_length = 4,
        max_enroll_length = 1
    ):

        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.max_length = max_length

        self.max_enroll_length = max_enroll_length
        self.visual_direc = visual_direc
        self.vsr_feat_dir = vsr_feat_dir
        

        if self.partition == 'train':
            self.aux_list =  open(aux_train_list).read().splitlines()[:]
            mix_lst = open(mix_lst_train_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_train_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            self.visual_codec_lst = open(visual_codec_lst_train_path).read().splitlines()
            self.visual_codec_dict =  {item.split()[0]: item for item in self.visual_codec_lst}

            self.vsr_sync_path_dict = open(vsr_sync_lst_train_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}

        else:
            self.aux_list =  open(aux_list).read().splitlines()
            mix_lst = open(mix_lst_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            self.vsr_sync_path_dict = open(vsr_sync_lst_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}
        sorted_mix_lst = mix_lst[:]
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end
        
    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
 
        mixtures = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        vsr_enrolls = []
        codec_lengths = []
        for line in batch_lst:
            target_uttid = line.split(' ')[0]
            mixture_path = line.split(' ')[1]
            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(
                    mixture, orig_sr=sr, target_sr=self.sampling_rate
                )

            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length * self.sampling_rate) - len(mixture)),
                )
            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)

           
            ### read target audio codec
            codec_path = self.codec_dict[target_uttid].split(' ')[1]
            codec = np.load(codec_path)
            if codec.shape[0]< int(min_length * self.codec_rate):
                pad_length = int(min_length * self.codec_rate) - codec.shape[0]
                codec = np.pad(codec, ((0, pad_length), (0, 0)), mode='constant')
            else:
                codec = codec[:int(min_length * self.codec_rate), :]
            codecs.append(codec)
            codec_lengths.append(int(min_length * self.codec_rate))

            if self.partition == 'train':
                
                visual_codec_path = self.visual_codec_dict[target_uttid].split(' ')[1]
                visual_codec = np.load(visual_codec_path)
                #!!!! for train_scale dataset, no visual token from avhubert, so use zero temporally/
                # visual_codecs.append(visual_codec)
                if visual_codec.shape[0]< int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - visual_codec.shape[0]
                    visual_codec = np.pad(visual_codec, (pad_length, 0), mode='constant')

                else:
                    visual_codec = visual_codec[:int(min_length * self.codec_rate)]
                visual_codecs.append(visual_codec)
            else:
                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                
            #=================================read vsr ================================#
                
            if self.partition == 'train':
                vsr_sync_path = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_feat = np.load(vsr_sync_path)
                if len(vsr_sync_feat.shape)==3:
                    vsr_sync_feat = vsr_sync_feat.squeeze(0)
                if vsr_sync_feat.shape[0] < int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    # 对第0维进行填充
                    vsr_sync_feat = np.pad(vsr_sync_feat, ((0, pad_length), (0, 0)), mode='constant')
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
            else:
                vsr_sync_path = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_feat = np.load(vsr_sync_path)
                if len(vsr_sync_feat.shape)==3:
                    vsr_sync_feat = vsr_sync_feat.squeeze(0)
                if vsr_sync_feat.shape[0] < int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    # 对第0维进行填充
                    vsr_sync_feat = np.pad(vsr_sync_feat, ((0, pad_length), (0, 0)), mode='constant')
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
    

            #==============================================================================

            # read enroll audio
            enroll_path = self.enroll_dict[target_uttid].split(' ')[1]
            enroll_audio, sr = audioread(enroll_path)
            if sr != self.sampling_rate:
                enroll_audio = librosa.resample(
                    enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                )
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(
                    enroll_audio,
                    (
                        0,
                        int(self.max_enroll_length * self.sampling_rate)
                        - len(enroll_audio),
                    ),
                )
            else:
                enroll_audio = enroll_audio[
                    0 : int(self.max_enroll_length * self.sampling_rate)
                ]
             
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)

        
        #     print("mixture mismatch1!!!")

        #     print("vsr_sync_feats mismatch1!!!")
            
        np_mixtures = np.asarray(mixtures)
        np_enrolls = np.asarray(enrolls)
        np_codecs = np.asarray(codecs)
        np_visual_codecs =  np.asarray(visual_codecs)
        np_visual_sync_feature =  np.asarray(vsr_sync_feats)
        np_codec_lengths = np.asarray(codec_lengths)
        return (
            np_mixtures,
             np_enrolls,
             np_codecs,
             np_visual_codecs,
            #  np_visual_enroll,
             np_visual_sync_feature,
             np_codec_lengths

             )
    # mixtures,enrolls,codecs,visual_codecs, visual_enroll, visual_sync_feature
    

    def __len__(self):
        return len(self.minibatch)
        # return 10 


class dataset_vocc_vox2(data.Dataset):
    def __init__(
        self,
        vsr_feat_dir,
        vsr_sync_lst_train_path,
        vsr_sync_lst_path,
        visual_direc,
        aux_list,
        aux_train_list,
        mix_lst_path,
        mix_lst_train_path,
        codec_lst_path, 
        codec_lst_train_path, 
        visual_codec_lst_train_path,
        batch_size,
        partition="val",
        sampling_rate=16000,
        codec_rate = 25,
        mix_no=2,
        max_length = 4,
        max_enroll_length = 1
    ):

        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.max_length = max_length

        self.max_enroll_length = max_enroll_length
        self.visual_direc = visual_direc
        self.vsr_feat_dir = vsr_feat_dir
        

        if self.partition == 'train':
            self.aux_list =  open(aux_train_list).read().splitlines()[:]
            mix_lst = open(mix_lst_train_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_train_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            self.visual_codec_lst = open(visual_codec_lst_train_path).read().splitlines()
            self.visual_codec_dict =  {item.split()[0]: item for item in self.visual_codec_lst}

            self.vsr_sync_path_dict = open(vsr_sync_lst_train_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}

        else:
            self.aux_list =  open(aux_list).read().splitlines()
            mix_lst = open(mix_lst_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            self.vsr_sync_path_dict = open(vsr_sync_lst_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}
        sorted_mix_lst = mix_lst[:]
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end
        
    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
 
        mixtures = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        vsr_enrolls = []
        for line in batch_lst:
            target_uttid = line.split(' ')[0]
            mixture_path = '_'.join(line.split(' ')[1].split('_')[:-6])+'.wav'
            vocc_info = line.split(' ')[1].split('_')[-6:-3]
            mask_start = vocc_info[0]
            mask_len = vocc_info[1]
            mask_type = vocc_info[2]

            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(
                    mixture, orig_sr=sr, target_sr=self.sampling_rate
                )

            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length * self.sampling_rate) - len(mixture)),
                )
            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)

           
            ### read target audio codec
            codec_path = self.codec_dict[target_uttid].split(' ')[1]
            codec = np.load(codec_path)
            if codec.shape[0]< int(min_length * self.codec_rate):
                pad_length = int(min_length * self.codec_rate) - codec.shape[0]
                codec = np.pad(codec, ((0, pad_length), (0, 0)), mode='constant')
            else:
                codec = codec[:int(min_length * self.codec_rate), :]
            codecs.append(codec)

            if self.partition == 'train':
                #!!!! for train_scale dataset, no visual token from avhubert, so use zero temporally/
                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                if visual_codec.shape[0]< int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - visual_codec.shape[0]
                    visual_codec = np.pad(visual_codec, ((0, pad_length), (0, 0)), mode='constant')
                else:
                    visual_codec = visual_codec[:int(min_length * self.codec_rate)]
                visual_codecs.append(visual_codec)
            else:
                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                
            #=================================read vsr ================================#
                
            if self.partition == 'train':
                vsr_sync_path_with_vocc = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_path = vsr_sync_path_with_vocc
                vocc_info_mask_type = vsr_sync_path_with_vocc.split('.npy')[0].split('/')[-1].split('_mask')[1].split('_')[0]
                vocc_info_mask_start = vsr_sync_path_with_vocc.split('.npy')[0].split('/')[-1].split('_start')[1].split('_')[0]
                vocc_info_mask_len = vsr_sync_path_with_vocc.split('.npy')[0].split('/')[-1].split('_len')[1].split('_')[0]
                
                assert [vocc_info_mask_type, vocc_info_mask_start, vocc_info_mask_len] == [mask_type, mask_start, mask_len]
                vsr_sync_feat = np.load(vsr_sync_path)
                if vsr_sync_feat.shape[0] < int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    # 对第0维进行填充
                    vsr_sync_feat = np.pad(vsr_sync_feat, ((0, pad_length), (0, 0)), mode='constant')
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
            else:
                vsr_sync_path = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_feat = np.load(vsr_sync_path)
                if vsr_sync_feat.shape[0] < int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    # 对第0维进行填充
                    vsr_sync_feat = np.pad(vsr_sync_feat, ((0, pad_length), (0, 0)), mode='constant')
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
    

            #==============================================================================

            # read enroll audio
            enroll_path = self.enroll_dict[target_uttid].split(' ')[1]
            enroll_audio, sr = audioread(enroll_path)
            if sr != self.sampling_rate:
                enroll_audio = librosa.resample(
                    enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                )
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(
                    enroll_audio,
                    (
                        0,
                        int(self.max_enroll_length * self.sampling_rate)
                        - len(enroll_audio),
                    ),
                )
            else:
                enroll_audio = enroll_audio[
                    0 : int(self.max_enroll_length * self.sampling_rate)
                ]
             
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)

        
        #     print("mixture mismatch1!!!")

        #     print("vsr_sync_feats mismatch1!!!")
            
        np_mixtures = np.asarray(mixtures)
        np_enrolls = np.asarray(enrolls)
        np_codecs = np.asarray(codecs)
        np_visual_codecs =  np.asarray(visual_codecs)
        np_visual_sync_feature =  np.asarray(vsr_sync_feats)
        return (
            np_mixtures,
             np_enrolls,
             np_codecs,
             np_visual_codecs,
            #  np_visual_enroll,
             np_visual_sync_feature,

             )
    # mixtures,enrolls,codecs,visual_codecs, visual_enroll, visual_sync_feature
    

    def __len__(self):
        return len(self.minibatch)
        # return 10 


class dataset_vocc_lrs3(data.Dataset):
    def __init__(
        self,
        vsr_feat_dir,
        vsr_sync_lst_train_path,
        vsr_sync_lst_path,
        visual_direc,
        aux_list,
        aux_train_list,
        mix_lst_path,
        mix_lst_train_path,
        codec_lst_path, 
        codec_lst_train_path, 
        visual_codec_lst_train_path,
        batch_size,
        partition="val",
        sampling_rate=16000,
        codec_rate = 25,
        mix_no=2,
        max_length = 4,
        max_enroll_length = 1
    ):

        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.max_length = max_length

        self.max_enroll_length = max_enroll_length
        self.visual_direc = visual_direc
        self.vsr_feat_dir = vsr_feat_dir
        

        if self.partition == 'train':
            self.aux_list =  open(aux_train_list).read().splitlines()[:]
            mix_lst = open(mix_lst_train_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_train_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            self.visual_codec_lst = open(visual_codec_lst_train_path).read().splitlines()
            self.visual_codec_dict =  {item.split()[0]: item for item in self.visual_codec_lst}

            self.vsr_sync_path_dict = open(vsr_sync_lst_train_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}

        else:
            self.aux_list =  open(aux_list).read().splitlines()
            mix_lst = open(mix_lst_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            self.vsr_sync_path_dict = open(vsr_sync_lst_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}
        sorted_mix_lst = mix_lst[:]
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end
        
    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
 
        mixtures = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        vsr_enrolls = []
        for line in batch_lst:
            target_uttid = line.split(' ')[0]
            mixture_path = '_'.join(line.split(' ')[1].split('_')[:-6])+'.wav'
            vocc_info = line.split(' ')[1].split('_')[-6:-3]
            mask_start = vocc_info[0]
            mask_len = vocc_info[1]
            mask_type = vocc_info[2]

            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(
                    mixture, orig_sr=sr, target_sr=self.sampling_rate
                )

            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length * self.sampling_rate) - len(mixture)),
                )
            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)

           
            ### read target audio codec
            codec_path = self.codec_dict[target_uttid].split(' ')[1]
            codec = np.load(codec_path)
            if codec.shape[0]< int(min_length * self.codec_rate):
                pad_length = int(min_length * self.codec_rate) - codec.shape[0]
                codec = np.pad(codec, ((0, pad_length), (0, 0)), mode='constant')
            else:
                codec = codec[:int(min_length * self.codec_rate), :]
            codecs.append(codec)

            if self.partition == 'train':
                #!!!! for train_scale dataset, no visual token from avhubert, so use zero temporally/
                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                #     # pad_length = int(min_length * self.codec_rate) - visual_codec.shape[0]
                #     # visual_codec = np.pad(visual_codec, ((0, pad_length), (0, 0)), mode='constant')
                # else:
                # visual_codecs.append(visual_codec)
            else:
                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                
            #=================================read vsr ================================#
                
            if self.partition == 'train':
                vsr_sync_path_with_vocc = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_path = vsr_sync_path_with_vocc
                vocc_info_mask_type = vsr_sync_path_with_vocc.split('.npy')[0].split('/')[-1].split('_mask')[1].split('_')[0]
                vocc_info_mask_start = vsr_sync_path_with_vocc.split('.npy')[0].split('/')[-1].split('_start')[1].split('_')[0]
                vocc_info_mask_len = vsr_sync_path_with_vocc.split('.npy')[0].split('/')[-1].split('_len')[1].split('_')[0]
                
                assert [vocc_info_mask_type, vocc_info_mask_start, vocc_info_mask_len] == [mask_type, mask_start, mask_len]
                vsr_sync_feat = np.load(vsr_sync_path)
                if vsr_sync_feat.shape[0] < int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    # 对第0维进行填充
                    vsr_sync_feat = np.pad(vsr_sync_feat, ((0, pad_length), (0, 0)), mode='constant')
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
            else:
                vsr_sync_path = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_feat = np.load(vsr_sync_path)
                if vsr_sync_feat.shape[0] < int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    # 对第0维进行填充
                    vsr_sync_feat = np.pad(vsr_sync_feat, ((0, pad_length), (0, 0)), mode='constant')
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
    

            #==============================================================================

            # read enroll audio
            enroll_path = self.enroll_dict[target_uttid].split(' ')[1]
            enroll_audio, sr = audioread(enroll_path)
            if sr != self.sampling_rate:
                enroll_audio = librosa.resample(
                    enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                )
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(
                    enroll_audio,
                    (
                        0,
                        int(self.max_enroll_length * self.sampling_rate)
                        - len(enroll_audio),
                    ),
                )
            else:
                enroll_audio = enroll_audio[
                    0 : int(self.max_enroll_length * self.sampling_rate)
                ]
             
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)

        
        #     print("mixture mismatch1!!!")

        #     print("vsr_sync_feats mismatch1!!!")
            
        np_mixtures = np.asarray(mixtures)
        np_enrolls = np.asarray(enrolls)
        np_codecs = np.asarray(codecs)
        np_visual_codecs =  np.asarray(visual_codecs)
        np_visual_sync_feature =  np.asarray(vsr_sync_feats)
        return (
            np_mixtures,
             np_enrolls,
             np_codecs,
             np_visual_codecs,
            #  np_visual_enroll,
             np_visual_sync_feature,

             )
    # mixtures,enrolls,codecs,visual_codecs, visual_enroll, visual_sync_feature
    

    def __len__(self):
        return len(self.minibatch)
        # return 10 


class dataset_memo(data.Dataset):
    def __init__(
        self,
        obj_dir,
        obj_mask_dir,
        speaker_dict,
        mix_lst_path,
        visual_direc,
        mixture_direc,
        batch_size,
        partition="val",
        sampling_rate=16000,
        mix_no=2,
        max_length=6,
    ):

        self.minibatch = []
        self.visual_direc = visual_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.speaker_id = speaker_dict
        self.max_length = max_length

        self.obj_dir = obj_dir
        self.obj_mask_dir = obj_mask_dir

        self.normMean = 0.4161
        self.normStd = 0.1688

        mix_lst = open(mix_lst_path).read().splitlines()
        mix_lst = list(filter(lambda x: x.split(",")[0] == partition, mix_lst))
        spk2utt_dict = {}
        for line in mix_lst:
            path_line = line.split(",")[0:-6]
            for i in range(2):
                ID = line.split(",")[i * 4 + 2]
                utt_path = (
                    self.mixture_direc
                    + self.partition
                    + "/s%d/" % (i + 1)
                    + ",".join(path_line).replace(",", "_").replace("/", "_")
                    + ".wav"
                )
                if ID not in spk2utt_dict:
                    spk2utt_dict[ID] = [utt_path]
                else:
                    spk2utt_dict[ID].append(utt_path)
        self.spk2utt_dict = spk2utt_dict

        sorted_mix_lst = sorted(
            mix_lst, key=lambda data: float(data.split(",")[-1]), reverse=True
        )
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)
        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
        for _ in range(len(batch_lst)):
            if float(batch_lst[_].split(",")[-7]) < min_length:
                min_length = float(batch_lst[_].split(",")[-7])

        enroll_length = 3

        mixtures = []
        audios = []
        enrolls = []
        visuals = []
        clean_visuals = []
        speakers = []
        mask_starts = []
        mask_lengths = []
        mask_types = []

        for line in batch_lst:
            path_line = line.split(",")[0:-6]
            mixture_path = (
                self.mixture_direc
                + self.partition
                + "/mix/"
                + ",".join(path_line).replace(",", "_").replace("/", "_")
                + ".wav"
            )
            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(
                    mixture, orig_sr=sr, target_sr=self.sampling_rate
                )

            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length * self.sampling_rate) - len(mixture)),
                )

            for c in range(self.C):
                # read target audio
                audio_path = (
                    self.mixture_direc
                    + self.partition
                    + "/s%d/" % (c + 1)
                    + ",".join(path_line).replace(",", "_").replace("/", "_")
                    + ".wav"
                )
                spk_id = line.split(",")[c * 4 + 2]
                if self.partition != "train":
                    enroll_path = self.spk2utt_dict[spk_id][0]
                else:
                    enroll_path = random.choice(self.spk2utt_dict[spk_id])

                audio, sr = audioread(audio_path)
                audio = audio[0 : int(min_length * self.sampling_rate)]
                if sr != self.sampling_rate:
                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=self.sampling_rate
                    )
                if len(audio) < int(min_length * self.sampling_rate):
                    audio = np.pad(
                        audio,
                        (0, int(min_length * self.sampling_rate) - len(audio)),
                    )
                audios.append(audio)

                enroll_audio, sr = audioread(enroll_path)
                if sr != self.sampling_rate:
                    enroll_audio = librosa.resample(
                        enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                    )
                if len(enroll_audio) < int(enroll_length * self.sampling_rate):
                    enroll_audio = np.pad(
                        enroll_audio,
                        (
                            0,
                            int(enroll_length * self.sampling_rate)
                            - len(enroll_audio),
                        ),
                    )
                else:
                    enroll_audio = enroll_audio[
                        0 : int(enroll_length * self.sampling_rate)
                    ]
                enrolls.append(enroll_audio)

                # read video
                mask_start = int(line.split(",")[c * 3 + 10])
                mask_length = int(line.split(",")[c * 3 + 11])
                mask_type = int(
                    line.split(",")[c * 3 + 12]
                )  # 0:full_mask 1: occluded 2: low resolution

                visual_path = (
                    self.visual_direc
                    + line.split(",")[1 + c * 4]
                    + "/"
                    + line.split(",")[2 + c * 4]
                    + "/"
                    + line.split(",")[3 + c * 4]
                    + ".mp4"
                )
                captureObj = cv.VideoCapture(visual_path)
                roiSequence = []
                clean_roiSequence = []
                roiSize = 112
                start = 0

                if mask_type == 1 and self.partition != "test":
                    occlude_img, occluder_img, occluder_mask = get_occluders(
                        self.obj_dir, self.obj_mask_dir, state=self.partition
                    )
                    alpha_mask = np.expand_dims(occluder_mask, axis=2)
                    alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
                elif mask_type == 1 and self.partition == "test":
                    occlude_img, occluder_img, occluder_mask = get_occluders(
                        self.obj_dir + "_test",
                        self.obj_mask_dir + "_test",
                        state=self.partition,
                    )
                    alpha_mask = np.expand_dims(occluder_mask, axis=2)
                    alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
                ratio = mask_length / (min_length * 25)
                while captureObj.isOpened():
                    ret, frame = captureObj.read()
                    if ret == True:
                        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        grayed = grayed / 255
                        grayed = cv.resize(grayed, (roiSize * 2, roiSize * 2))
                        roi = grayed[
                            int(roiSize - (roiSize / 2)) : int(
                                roiSize + (roiSize / 2)
                            ),
                            int(roiSize - (roiSize / 2)) : int(
                                roiSize + (roiSize / 2)
                            ),
                        ]
                        clean_roiSequence.append(roi)
                        if (
                            start >= mask_start
                            and start < mask_start + mask_length
                        ):
                            if mask_type == 0:
                                roi = np.zeros_like(roi)
                            elif mask_type == 1:
                                frame = cv.resize(
                                    frame, (roiSize * 2, roiSize * 2)
                                )
                                roi = frame[
                                    int(roiSize - (roiSize / 2)) : int(
                                        roiSize + (roiSize / 2)
                                    ),
                                    int(roiSize - (roiSize / 2)) : int(
                                        roiSize + (roiSize / 2)
                                    ),
                                ]
                                if self.partition == "train":
                                    offset_x = random.uniform(13, 17)
                                    offset_y = random.uniform(13, 17)
                                    x = random.uniform(112 / 2 - 5, 112 / 2 + 5)
                                    y = random.uniform(112 / 2 - 5, 112 / 2 + 5)
                                else:
                                    offset_x = 15
                                    offset_y = 15
                                    x = 112 / 2
                                    y = 112 / 2
                                roi = overlay_image_alpha(
                                    roi,
                                    occluder_img,
                                    int(x - offset_x),
                                    int(y - offset_y),
                                    alpha_mask
                                )
                                roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                                # cv.imwrite('frames/'+'org_image_'+str(id_)+'.png',roi)
                                # pdb.set_trace()
                                roi = roi / 255
                            elif mask_type == 2:
                                frame = cv.resize(
                                    frame, (roiSize * 2, roiSize * 2)
                                )
                                roi = frame[
                                    int(roiSize - (roiSize / 2)) : int(
                                        roiSize + (roiSize / 2)
                                    ),
                                    int(roiSize - (roiSize / 2)) : int(
                                        roiSize + (roiSize / 2)
                                    ),
                                ]
                                if self.partition == "train":
                                    if random.random() < 0.5:
                                        var = random.uniform(0.02, 0.2)
                                        roi = random_noise(
                                                roi,
                                                mode="gaussian",
                                                mean=0,
                                                var=var,
                                                clip=True,
                                            )* 255
                    
                                        roi = np.uint8(roi)
                                    else:
                                        blur = (
                                            torchvision.transforms.GaussianBlur(
                                                kernel_size=(13, 13),
                                                sigma=(4, 8),
                                            )
                                        )
                                        roi = (
                                            blur(
                                                torch.tensor(roi)
                                                .unsqueeze(0)
                                                .permute(0, 3, 1, 2)
                                            )
                                            .permute(0, 2, 3, 1)
                                            .squeeze(0)
                                            .numpy()
                                        )
                                else:
                                    times = 10
                                    roi = cv.resize(
                                        roi,
                                        (roiSize // times, roiSize // times),
                                    )
                                    roi = cv.resize(roi, (roiSize, roiSize))
                                roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                                # cv.imwrite('frames/'+'org_image_'+str(id_)+'.png',roi)
                                roi = roi / 255
                            else:
                                sys.exit("error: ", mask_type)
                        roiSequence.append(roi)
                        start += 1
                    else:
                        break
                captureObj.release()
                visual = np.asarray(roiSequence)
                visual = visual[0 : int(min_length * self.fps), ...]
                visual = (visual - self.normMean) / self.normStd
                if visual.shape[0] < int(min_length * self.fps):
                    visual = np.pad(
                        visual,
                        (
                            (0, int(min_length * self.fps) - visual.shape[0]),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="edge",
                    )
                visuals.append(visual)

                clean_visual = np.asarray(clean_roiSequence)
                clean_visual = clean_visual[0 : int(min_length * self.fps), ...]
                clean_visual = (clean_visual - self.normMean) / self.normStd
                if clean_visual.shape[0] < int(min_length * self.fps):
                    clean_visual = np.pad(
                        clean_visual,
                        (
                            (
                                0,
                                int(min_length * self.fps)
                                - clean_visual.shape[0],
                            ),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="edge",
                    )
                clean_visuals.append(clean_visual)

                # read speaker label
                speakers.append(self.speaker_id[line.split(",")[c * 4 + 2]])
                
                mask_types.append(mask_type)
                mask_starts.append(mask_start)
                mask_lengths.append(mask_length)
            mixtures.append(mixture)
            mixtures.append(mixture)
           
        np_mixtures = np.asarray(mixtures)
        np_audios = np.asarray(audios)
        np_visuals = np.asarray(visuals)
        np_speakers = np.asarray(speakers)
        np_enrolls = np.asarray(enrolls)
        np_clean_visuals = np.asarray(clean_visuals)

        return (
            np_mixtures,
            np_audios,
            np_visuals,
            np_clean_visuals,
            np_enrolls,
            np_speakers,)
            # mask_starts,
            # mask_lengths,
            # mask_types,

    def __len__(self):
        return len(self.minibatch)
        # return 10 


class dataset_lrs3_switch(data.Dataset):
    def __init__(
        self,
       mix_csv_path,
       codec_dir,
       visual_dir,
       switch_audio_dir,
        batch_size,
        partition="val",
        sampling_rate=16000,
        codec_rate = 25,
        mix_no=2,
        max_length = 6,
        max_enroll_length = 1,
        visual_frontend_ckpt='/mnt/users/hccl.local/wwu/MuSE/pretrain_networks/visual_frontend.pt',
        occluder_asset_dir='/mnt/users/hccl.local/wwu/SEMO-621/data_preparation/Asset',
    ):
        
        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.max_length = max_length

        self.max_enroll_length = max_enroll_length
        self.visual_dir = visual_dir
        self.visual_corrupt = False
        self.switch_audio_dir = switch_audio_dir
        self.codec_dir = codec_dir


        self.visual_frontend_ckpt = visual_frontend_ckpt
        self.occluder_asset_dir = occluder_asset_dir

        self.vf = VisualFrontend()
        self.vf.load_state_dict(torch.load(self.visual_frontend_ckpt))

        mix_lst = open(mix_csv_path).read().splitlines()
        mix_lst = list(filter(lambda x: x.split(",")[0] == partition, mix_lst))[:]
        spk2utt_dict = {}
        for line in mix_lst:
            for i in range(2):
                ID = line.split(",")[i * 4 + 2]
                utt_path = ( switch_audio_dir + self.partition + "/s%d/" % (i + 1) +line.replace(",", "_").replace("/", "_") + ".wav")
                if ID not in spk2utt_dict:
                    spk2utt_dict[ID] = [utt_path]
                else:
                    spk2utt_dict[ID].append(utt_path)
        self.spk2utt_dict = spk2utt_dict

        sorted_mix_lst = sorted(
            mix_lst, key=lambda data: float(data.split(",")[-1]), reverse=True
        )
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)
        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end


    def process_video_with_mask(self, videoFile, test_ds_rate, mask_type, mask_start, mask_length, split, params):
        """
        处理带有mask的视频并提取特征
        """
        roiSize = params["roiSize"]
        normMean = params["normMean"]
        normStd = params["normStd"]
        vf = params["vf"]
        
        # 获取遮挡物（如果是部分遮挡）
        occluder_img = None
        alpha_mask = None
        if mask_type == 1:
            obj_dir = os.path.join(self.occluder_asset_dir, 'object_image_sr')
            obj_mask_dir = os.path.join(self.occluder_asset_dir, 'object_mask_x4')
            _, occluder_img, occluder_mask = get_occluders(obj_dir, obj_mask_dir, state = split)
            alpha_mask = np.expand_dims(occluder_mask, axis=2)
            alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
        
        # 读取视频并应用mask
        captureObj = cv.VideoCapture(videoFile)
        roiSequence = []
        roiSequence_vclean = []
        frame_idx = 0
        
        while captureObj.isOpened():
            ret, frame = captureObj.read()
            if not ret:
                break
            
            # 提取基础ROI
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = grayed / 255
            grayed = cv.resize(grayed, (roiSize * 2, roiSize * 2))
            base_roi = grayed[
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            ]
            if mask_type == -1:
                roiSequence_vclean.append(base_roi)
                frame_idx += 1
            # 应用mask
            else:
                masked_roi = apply_mask_to_frame(
                    base_roi, test_ds_rate, mask_type, frame, roiSize, split, 
                    mask_start, mask_length, frame_idx, occluder_img, alpha_mask
                )
                
                roiSequence.append(masked_roi)
                frame_idx += 1
        
        captureObj.release()
        
        # 提取视觉特征
        if mask_type!= -1 and len(roiSequence) > 0:
            inp = np.stack(roiSequence, axis=0)
            inp = np.expand_dims(inp, axis=[1, 2])
            inp = (inp - normMean) / normStd
            inputBatch = torch.from_numpy(inp).float() 
            
            vf.eval()
            with torch.no_grad():
                outputBatch = vf(inputBatch)
            out = torch.squeeze(outputBatch, dim=1) 
            return out
        else:

            inp = np.stack(roiSequence_vclean, axis=0)
            inp = np.expand_dims(inp, axis=[1, 2])
            inp = (inp - normMean) / normStd
            inputBatch = torch.from_numpy(inp).float() 
            
            vf.eval()
            with torch.no_grad():
                outputBatch = vf(inputBatch)
            out = torch.squeeze(outputBatch, dim=1) 
            return out

            
            # 保存特征
            # np.save(outputFile, visual_features)


    def apply_mask_to_frame(self, roi, test_ds_rate, mask_type, original_frame, roiSize, partition, mask_start, mask_length, frame_idx, 
                        occluder_img=None, alpha_mask=None):
        """
        应用不同类型的mask到帧上
        
        Args:
            roi: 当前帧的ROI区域
            mask_type: 遮挡类型 (0:full_mask, 1:occluded, 2:low resolution)
            original_frame: 原始帧
            roiSize: ROI尺寸
            partition: 数据集分区 (train/val/test)
            mask_start: mask开始帧
            mask_length: mask持续时间
            frame_idx: 当前帧索引
            occluder_img: 遮挡物图像
            alpha_mask: 遮挡物遮罩
        """
        import random
        import torchvision
        from skimage.util import random_noise
        import sys
        
        # 如果不在mask范围内，直接返回原始ROI
        if frame_idx < mask_start or frame_idx >= mask_start + mask_length:
            return roi
        
        if mask_type == 0:  # full_mask (全遮挡)
            return np.zeros_like(roi)
        
        elif mask_type == 1:  # occluded (部分遮挡)
            if occluder_img is None or alpha_mask is None:
                # 如果没有提供遮挡物，使用默认的全遮挡
                return np.zeros_like(roi)
            
            frame_resized = cv.resize(original_frame, (roiSize * 2, roiSize * 2))
            roi_color = frame_resized[
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            ]
            
            if partition != "test":
                offset_x = random.uniform(13, 17)
                offset_y = random.uniform(13, 17)
                x = random.uniform(roiSize / 2 - 5, roiSize / 2 + 5)
                y = random.uniform(roiSize / 2 - 5, roiSize / 2 + 5)
            else:
                offset_x = 15
                offset_y = 15
                x = roiSize / 2
                y = roiSize / 2
            
            # 应用遮挡
            roi_color = overlay_image_alpha(
                roi_color,
                occluder_img,
                int(x - offset_x),
                int(y - offset_y),
                alpha_mask,
            )
            roi_masked = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
            return roi_masked / 255
        
        elif mask_type == 2:  # low resolution (低分辨率)
            frame_resized = cv.resize(original_frame, (roiSize * 2, roiSize * 2))
            roi_color = frame_resized[
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            ]
            
            if partition != "test":
                if random.random() < 0.5:
                    # 添加高斯噪声
                    var = random.uniform(0.02, 0.2)
                    roi_color = random_noise(
                        roi_color, mode="gaussian", mean=0, var=var, clip=True
                    ) * 255
                    roi_color = np.uint8(roi_color)
                else:
                    # 高斯模糊
                    blur = torchvision.transforms.GaussianBlur(
                        kernel_size=(13, 13), sigma=(4, 8)
                    )
                    roi_tensor = torch.tensor(roi_color).unsqueeze(0).permute(0, 3, 1, 2)
                    roi_blurred = blur(roi_tensor).permute(0, 2, 3, 1).squeeze(0).numpy()
                    roi_color = roi_blurred
            else:
                # 测试时使用固定的低分辨率处理
                times = test_ds_rate
                roi_color = cv.resize(roi_color, (roiSize // times, roiSize // times))
                roi_color = cv.resize(roi_color, (roiSize, roiSize))
            
            roi_masked = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
            return roi_masked / 255
        
        else:
            sys.exit("error: unknown mask type", mask_type)
    
 
    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
 
        mixtures = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        vsr_enrolls = []
        switch_times = []
        for line in batch_lst:
            target_uttid = (line.split(',')[2] + '_' + line.split(',')[5]).replace('/','_')

            if self.visual_corrupt:
                vocc_info = line.split(' ')[1].split('_')[-6:-3]
                mask_start = vocc_info[0]
                mask_len = vocc_info[1]
                mask_type = vocc_info[2]
            mixture_path = ( self.switch_audio_dir + self.partition + "/mix/" + line.replace(",", "_").replace("/", "_") + ".wav" )
            
            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(mixture, orig_sr=sr, target_sr=self.sampling_rate)

            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(mixture,(0, int(min_length * self.sampling_rate) - len(mixture)),)

            enroll_audio = mixture[0 : int(self.max_enroll_length * self.sampling_rate)]
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(enroll_audio,(0, int(self.max_enroll_length * self.sampling_rate) - len(enroll_audio)),)
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)

            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)

            # enrolls.append(mixture)
             

            ### read target audio codec
            codec_path = self.codec_dir + self.partition + '/' + line.replace(",", "_").replace("/", "_") + ".npy"
            codec = np.load(codec_path)
            if codec.shape[0]< int(min_length * self.codec_rate):
                pad_length = int(min_length * self.codec_rate) - codec.shape[0]
                codec = np.pad(codec, ((0, pad_length), (0, 0)), mode='constant')
            else:
                codec = codec[:int(min_length * self.codec_rate), :]

            codecs.append(codec)

            # # import pdb;pdb.set_trace()
                
            #=================================read vsr ================================#
                
            tgt1_video_path = self.visual_dir + line.split(',')[1] + '/' + line.split(',')[2] + '.mp4'
            tgt2_video_path = self.visual_dir + line.split(',')[4] + '/' + line.split(',')[5] + '.mp4'
            tgt3_video_path = self.visual_dir + line.split(',')[7] + '/' + line.split(',')[8] + '.mp4'
            switch_time = float(line.split(',')[-1])
            switch_times.append(switch_time)


            test_ds_rate = 25
            mask_start = 99999
            mask_len = 99999
            mask_type = -1
            split = self.partition

             # 设置参数
            params = {
                "roiSize": 112, 
                "normMean": 0.4161, 
                "normStd": 0.1688, 
                "obj_dir": 1,
                "obj_mask_dir": 1,
                "vf" : self.vf
            }


            tgt1_visual_sync =  self.process_video_with_mask(tgt1_video_path, test_ds_rate, mask_type, mask_start, mask_len, split, params)
            tgt2_visual_sync =  self.process_video_with_mask(tgt2_video_path, test_ds_rate, mask_type, mask_start, mask_len, split, params)
            if switch_time > tgt1_visual_sync.shape[0]:
                print('tgt1_video_path; expect_switch, real_lenghth:', switch_time, tgt1_visual_sync.shape[0])
                 
                    # 拼接两个视觉特征
            tgt_switch_visual = np.concatenate((tgt1_visual_sync[:int(switch_time*25)], tgt2_visual_sync), axis=0)

            # 截断到最大长度
            tgt_switch_visual = tgt_switch_visual[:int(self.max_length * 25)]

            # 检查并padding
            current_length = tgt_switch_visual.shape[0]
            target_length = int(self.max_length * 25)

            if current_length < target_length:
                print("current len: ", current_length)
                pad_length = target_length - current_length
                # 只在第0维度（时间维度）进行padding
                pad_width = [(0, pad_length)] + [(0, 0)] * (len(tgt_switch_visual.shape) - 1)
                tgt_switch_visual = np.pad(tgt_switch_visual, pad_width, mode='constant', constant_values=0)
            
           
            # 添加到特征列表中
            vsr_sync_feats.append(tgt_switch_visual)
 
           
            # # vsr_sync_feat = vsr_sync_feat.squeeze(0)
            # # import pdb;pdb.set_trace()
            #     # import pdb; pdb.set_trace()
            #     # 对第0维进行填充
            #     # pad_length = int(min_length * self.codec_rate) - visual_codec.shape[0]
            #     # visual_codec = np.pad(visual_codec, ((0, pad_length), (0, 0)), mode='constant')
            # else:
            # vsr_sync_feats.append(vsr_sync_feat)

    
        np_mixtures = np.asarray(mixtures)
        np_enrolls = np.asarray(enrolls)
        np_codecs = np.asarray(codecs)
        np_visual_codecs =  np.asarray(codecs)[:,:,0]
        np_visual_sync_feature =  np.asarray(vsr_sync_feats)
        return (
            np_mixtures,
             np_enrolls,
             np_codecs,
             np_visual_codecs,
             np_visual_sync_feature,
             switch_times

             )
    # mixtures,enrolls,codecs,visual_codecs, visual_enroll, visual_sync_feature
    

    def __len__(self):
        return len(self.minibatch)
        # return 10 


class dataset_lrs3_switch_sws_token(data.Dataset):
    def __init__(
        self,
       mix_csv_path,
       codec_dir,
       visual_dir,
       switch_audio_dir,
        batch_size,
        partition="val",
        sampling_rate=16000,
        codec_rate = 25,
        mix_no=2,
        max_length = 6,
        max_enroll_length = 1,
        visual_frontend_ckpt='/mnt/users/hccl.local/wwu/MuSE/pretrain_networks/visual_frontend.pt',
        occluder_asset_dir='/mnt/users/hccl.local/wwu/SEMO-621/data_preparation/Asset',
    ):
        
        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.max_length = max_length

        self.max_enroll_length = max_enroll_length
        self.visual_dir = visual_dir
        self.visual_corrupt = False
        self.switch_audio_dir = switch_audio_dir
        self.codec_dir = codec_dir

        self.visual_frontend_ckpt = visual_frontend_ckpt
        self.occluder_asset_dir = occluder_asset_dir

        self.vf = VisualFrontend()
        self.vf.load_state_dict(torch.load(self.visual_frontend_ckpt))

        mix_lst = open(mix_csv_path).read().splitlines()
        mix_lst = list(filter(lambda x: x.split(",")[0] == partition, mix_lst))[:]
        spk2utt_dict = {}
        
        for line in mix_lst:
            for i in range(2):
                ID = line.split(",")[i * 4 + 2]
                utt_path = ( switch_audio_dir + self.partition + "/s%d/" % (i + 1) +line.replace(",", "_").replace("/", "_") + ".wav")
                if ID not in spk2utt_dict:
                    spk2utt_dict[ID] = [utt_path]
                else:
                    spk2utt_dict[ID].append(utt_path)
        self.spk2utt_dict = spk2utt_dict

        self.sws = 1025  

        sorted_mix_lst = sorted(
            mix_lst, key=lambda data: float(data.split(",")[-1]), reverse=True
        )
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)
        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end


    def process_video_with_mask(self, videoFile, test_ds_rate, mask_type, mask_start, mask_length, split, params):
        """
        处理带有mask的视频并提取特征
        """
        roiSize = params["roiSize"]
        normMean = params["normMean"]
        normStd = params["normStd"]
        vf = params["vf"]
        
        # 获取遮挡物（如果是部分遮挡）
        occluder_img = None
        alpha_mask = None
        if mask_type == 1:
            obj_dir = os.path.join(self.occluder_asset_dir, 'object_image_sr')
            obj_mask_dir = os.path.join(self.occluder_asset_dir, 'object_mask_x4')
            _, occluder_img, occluder_mask = get_occluders(obj_dir, obj_mask_dir, state = split)
            alpha_mask = np.expand_dims(occluder_mask, axis=2)
            alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
        
        # 读取视频并应用mask
        captureObj = cv.VideoCapture(videoFile)
        roiSequence = []
        roiSequence_vclean = []
        frame_idx = 0
        
        while captureObj.isOpened():
            ret, frame = captureObj.read()
            if not ret:
                break
            
            # 提取基础ROI
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = grayed / 255
            grayed = cv.resize(grayed, (roiSize * 2, roiSize * 2))
            base_roi = grayed[
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            ]
            if mask_type == -1:
                roiSequence_vclean.append(base_roi)
                frame_idx += 1
            # 应用mask
            else:
                masked_roi = apply_mask_to_frame(
                    base_roi, test_ds_rate, mask_type, frame, roiSize, split, 
                    mask_start, mask_length, frame_idx, occluder_img, alpha_mask
                )
                
                roiSequence.append(masked_roi)
                frame_idx += 1
        
        captureObj.release()
        
        # 提取视觉特征
        if mask_type!= -1 and len(roiSequence) > 0:
            inp = np.stack(roiSequence, axis=0)
            inp = np.expand_dims(inp, axis=[1, 2])
            inp = (inp - normMean) / normStd
            inputBatch = torch.from_numpy(inp).float() 
            
            vf.eval()
            with torch.no_grad():
                outputBatch = vf(inputBatch)
            out = torch.squeeze(outputBatch, dim=1) 
            return out
        else:

            inp = np.stack(roiSequence_vclean, axis=0)
            inp = np.expand_dims(inp, axis=[1, 2])
            inp = (inp - normMean) / normStd
            inputBatch = torch.from_numpy(inp).float() 
            
            vf.eval()
            with torch.no_grad():
                outputBatch = vf(inputBatch)
            out = torch.squeeze(outputBatch, dim=1) 
            return out

            
            # 保存特征
            # np.save(outputFile, visual_features)


    def apply_mask_to_frame(self, roi, test_ds_rate, mask_type, original_frame, roiSize, partition, mask_start, mask_length, frame_idx, 
                        occluder_img=None, alpha_mask=None):
        """
        应用不同类型的mask到帧上
        
        Args:
            roi: 当前帧的ROI区域
            mask_type: 遮挡类型 (0:full_mask, 1:occluded, 2:low resolution)
            original_frame: 原始帧
            roiSize: ROI尺寸
            partition: 数据集分区 (train/val/test)
            mask_start: mask开始帧
            mask_length: mask持续时间
            frame_idx: 当前帧索引
            occluder_img: 遮挡物图像
            alpha_mask: 遮挡物遮罩
        """
        import random
        import torchvision
        from skimage.util import random_noise
        import sys
        
        # 如果不在mask范围内，直接返回原始ROI
        if frame_idx < mask_start or frame_idx >= mask_start + mask_length:
            return roi
        
        if mask_type == 0:  # full_mask (全遮挡)
            return np.zeros_like(roi)
        
        elif mask_type == 1:  # occluded (部分遮挡)
            if occluder_img is None or alpha_mask is None:
                # 如果没有提供遮挡物，使用默认的全遮挡
                return np.zeros_like(roi)
            
            frame_resized = cv.resize(original_frame, (roiSize * 2, roiSize * 2))
            roi_color = frame_resized[
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            ]
            
            if partition != "test":
                offset_x = random.uniform(13, 17)
                offset_y = random.uniform(13, 17)
                x = random.uniform(roiSize / 2 - 5, roiSize / 2 + 5)
                y = random.uniform(roiSize / 2 - 5, roiSize / 2 + 5)
            else:
                offset_x = 15
                offset_y = 15
                x = roiSize / 2
                y = roiSize / 2
            
            # 应用遮挡
            roi_color = overlay_image_alpha(
                roi_color,
                occluder_img,
                int(x - offset_x),
                int(y - offset_y),
                alpha_mask,
            )
            roi_masked = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
            return roi_masked / 255
        
        elif mask_type == 2:  # low resolution (低分辨率)
            frame_resized = cv.resize(original_frame, (roiSize * 2, roiSize * 2))
            roi_color = frame_resized[
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
                int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            ]
            
            if partition != "test":
                if random.random() < 0.5:
                    # 添加高斯噪声
                    var = random.uniform(0.02, 0.2)
                    roi_color = random_noise(
                        roi_color, mode="gaussian", mean=0, var=var, clip=True
                    ) * 255
                    roi_color = np.uint8(roi_color)
                else:
                    # 高斯模糊
                    blur = torchvision.transforms.GaussianBlur(
                        kernel_size=(13, 13), sigma=(4, 8)
                    )
                    roi_tensor = torch.tensor(roi_color).unsqueeze(0).permute(0, 3, 1, 2)
                    roi_blurred = blur(roi_tensor).permute(0, 2, 3, 1).squeeze(0).numpy()
                    roi_color = roi_blurred
            else:
                # 测试时使用固定的低分辨率处理
                times = test_ds_rate
                roi_color = cv.resize(roi_color, (roiSize // times, roiSize // times))
                roi_color = cv.resize(roi_color, (roiSize, roiSize))
            
            roi_masked = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
            return roi_masked / 255
        
        else:
            sys.exit("error: unknown mask type", mask_type)
    
        
    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
 
        mixtures = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        vsr_enrolls = []
        switch_times = []
        for line in batch_lst:
            target_uttid = (line.split(',')[2] + '_' + line.split(',')[5]).replace('/','_')

            if self.visual_corrupt:
                vocc_info = line.split(' ')[1].split('_')[-6:-3]
                mask_start = vocc_info[0]
                mask_len = vocc_info[1]
                mask_type = vocc_info[2]
            mixture_path = ( self.switch_audio_dir + self.partition + "/mix/" + line.replace(",", "_").replace("/", "_") + ".wav" )
            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(mixture, orig_sr=sr, target_sr=self.sampling_rate)
            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(mixture,(0, int(min_length * self.sampling_rate) - len(mixture)),)
            enroll_audio = mixture[0 : int(self.max_enroll_length * self.sampling_rate)]
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(enroll_audio,(0, int(self.max_enroll_length * self.sampling_rate) - len(enroll_audio)),)
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)
            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)
            # enrolls.append(mixture)
             
            codec_path = self.codec_dir + self.partition + '/' + line.replace(",", "_").replace("/", "_") + ".npy"
            codec = np.load(codec_path)
            if codec.shape[0]< int(min_length * self.codec_rate):
                pad_length = int(min_length * self.codec_rate) - codec.shape[0]
                codec = np.pad(codec, ((0, pad_length), (0, 0)), mode='constant')
            else:
                codec = codec[:int(min_length * self.codec_rate), :]

            #=================================insert sws_token ================================#
            switch_time = float(line.split(',')[-1])
            switch_frame = int(switch_time * 25)
            switch_times.append(switch_time)
             # ---- codec 插入 sws code ----
            # codec 最后一维是 pred_nq=2，这里简单都填 self.sws
            # 一整帧 sws，所有 quantizer 都填 self.sws
            sws_code = np.full((1, codec.shape[1]),            # (1, N_q)
                fill_value=self.sws, # sws 的离散 id
                dtype=codec.dtype,
            )

            codec_new = np.concatenate(
                (codec[:switch_frame, :], sws_code, codec[switch_frame:, :],), axis=0,)                             
        
            codecs.append(codec_new)

            # # import pdb;pdb.set_trace()
             
                
            #=================================read vsr ================================#
                
            tgt1_video_path = self.visual_dir + line.split(',')[1] + '/' + line.split(',')[2] + '.mp4'
            tgt2_video_path = self.visual_dir + line.split(',')[4] + '/' + line.split(',')[5] + '.mp4'
            tgt3_video_path = self.visual_dir + line.split(',')[7] + '/' + line.split(',')[8] + '.mp4'
             

            test_ds_rate = 25
            mask_start = 99999
            mask_len = 99999
            mask_type = -1
            split = self.partition
             # 设置参数
            params = {
                "roiSize": 112, 
                "normMean": 0.4161, 
                "normStd": 0.1688, 
                "obj_dir": 1,
                "obj_mask_dir": 1,
                "vf" : self.vf
            }


            tgt1_visual_sync =  self.process_video_with_mask(tgt1_video_path, test_ds_rate, mask_type, mask_start, mask_len, split, params)
            tgt2_visual_sync =  self.process_video_with_mask(tgt2_video_path, test_ds_rate, mask_type, mask_start, mask_len, split, params)
            if switch_time > tgt1_visual_sync.shape[0]:
                print('tgt1_video_path; expect_switch, real_lenghth:', switch_time, tgt1_visual_sync.shape[0])
                 
             
                    # 拼接两个视觉特征
            tgt_switch_visual = np.concatenate((tgt1_visual_sync[:int(switch_time*25)], tgt2_visual_sync), axis=0)
            # 截断到最大长度
            tgt_switch_visual = tgt_switch_visual[:int(self.max_length * 25)]
            # 检查并padding
            current_length = tgt_switch_visual.shape[0]
            target_length = int(self.max_length * 25)

            if current_length < target_length:
                print("current len: ", current_length)
                pad_length = target_length - current_length
                # 只在第0维度（时间维度）进行padding
                pad_width = [(0, pad_length)] + [(0, 0)] * (len(tgt_switch_visual.shape) - 1)
                tgt_switch_visual = np.pad(tgt_switch_visual, pad_width, mode='constant', constant_values=0)
            
            #=================================insert vsr sws token ================================#
            v_sync_switch_frame = tgt_switch_visual[switch_frame : switch_frame + 1, :]  # (1, D_v)
            tgt_switch_visual_new = np.concatenate((tgt_switch_visual[:switch_frame, :], v_sync_switch_frame, tgt_switch_visual[switch_frame:, :]), axis=0)                                   
            # 添加到特征列表中
            vsr_sync_feats.append(tgt_switch_visual_new)
 
           
        np_mixtures = np.asarray(mixtures)
        np_enrolls = np.asarray(enrolls)
        np_codecs = np.asarray(codecs)
        np_visual_codecs =  np.asarray(codecs)[:,:,0]
        np_visual_sync_feature =  np.asarray(vsr_sync_feats)
        return (
            np_mixtures,
             np_enrolls,
             np_codecs,
             np_visual_codecs,
             np_visual_sync_feature,
             switch_times

             )
             
    # mixtures,enrolls,codecs,visual_codecs, visual_enroll, visual_sync_feature
    

    def __len__(self):
        return len(self.minibatch)
        # return 10 


class dataset_scale(data.Dataset):
    def __init__(
        self,
        vsr_feat_dir,
        vsr_sync_lst_train_path,
        vsr_sync_lst_path,
        visual_direc,
        aux_list,
        aux_train_list,
        mix_lst_path,
        mix_lst_train_path,
        codec_lst_path, 
        codec_lst_train_path, 
        visual_codec_lst_train_path,
        batch_size,
        partition="val",
        sampling_rate=16000,
        codec_rate = 25,
        mix_no=2,
        max_length = 4,
        max_enroll_length = 1
    ):

        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.max_length = max_length

        self.max_enroll_length = max_enroll_length
        self.visual_direc = visual_direc
        self.vsr_feat_dir = vsr_feat_dir
        
        if self.partition == 'train':
            self.aux_list =  open(aux_train_list).read().splitlines()[:]
            mix_lst = open(mix_lst_train_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_train_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}


            self.vsr_sync_path_dict = open(vsr_sync_lst_train_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}

        else:
            self.aux_list =  open(aux_list).read().splitlines()
            mix_lst = open(mix_lst_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            self.vsr_sync_path_dict = open(vsr_sync_lst_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}
        sorted_mix_lst = mix_lst[:]
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)
        

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end
        
    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
 
        mixtures = []
        tgts = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        vsr_enrolls = []
        for line in batch_lst:
            target_uttid = line.split(' ')[0]
            mixture_path = line.split(' ')[1]
            tgt_path = '/'.join(mixture_path.split('/')[:-2]) + '/s1/' + mixture_path.split('/')[-1]
            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(
                    mixture, orig_sr=sr, target_sr=self.sampling_rate
                )
            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length * self.sampling_rate) - len(mixture)),
                )
            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)

            tgt, sr = audioread(tgt_path)
            if sr != self.sampling_rate:
                tgt = librosa.resample(tgt, orig_sr=sr, target_sr=self.sampling_rate)
            # truncate
            tgt = tgt[0 : int(min_length * self.sampling_rate)]
            if len(tgt) < int(min_length * self.sampling_rate):
                tgt = np.pad(tgt,(0, int(min_length * self.sampling_rate) - len(tgt)),)
            tgt = self.mel_proc.mel_one_np(tgt)
            tgts.append(tgt)

           
            ### read target audio codec
            codec_path = self.codec_dict[target_uttid].split(' ')[1]
            codec = np.load(codec_path)
            if codec.shape[0]< int(min_length * self.codec_rate):
                pad_length = int(min_length * self.codec_rate) - codec.shape[0]
                codec = np.pad(codec, ((0, pad_length), (0, 0)), mode='constant')
            else:
                codec = codec[:int(min_length * self.codec_rate), :]
            codecs.append(codec)

            if self.partition == 'train':

                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                if visual_codec.shape[0]< int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - visual_codec.shape[0]
                    visual_codec = np.pad(visual_codec, (0, pad_length), mode='constant')
                else:
                    visual_codec = visual_codec[:int(min_length * self.codec_rate)]
                visual_codecs.append(visual_codec)
            else:
                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                
            #=================================read vsr ================================#
                
            if self.partition == 'train':
                vsr_sync_path = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_feat = np.load(vsr_sync_path)
                if vsr_sync_feat.shape[0]< int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    vsr_sync_feat = np.pad(vsr_sync_feat, (0, pad_length), mode='constant')
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
            else:
                vsr_sync_path = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_feat = np.load(vsr_sync_path)
                if vsr_sync_feat.shape[0]< int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    vsr_sync_feat = np.pad(vsr_sync_feat, (0, pad_length), mode='constant')
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
    

            #==============================================================================

            # read enroll audio
            enroll_path = self.enroll_dict[target_uttid].split(' ')[1]
            enroll_audio, sr = audioread(enroll_path)
            if sr != self.sampling_rate:
                enroll_audio = librosa.resample(
                    enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                )
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(
                    enroll_audio,
                    (
                        0,
                        int(self.max_enroll_length * self.sampling_rate)
                        - len(enroll_audio),
                    ),
                )
            else:
                enroll_audio = enroll_audio[
                    0 : int(self.max_enroll_length * self.sampling_rate)
                ]
             
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)

 
        np_mixtures = np.asarray(mixtures)
        np_enrolls = np.asarray(enrolls)
        np_codecs = np.asarray(codecs)
        np_visual_codecs =  np.asarray(visual_codecs)
        np_visual_sync_feature =  np.asarray(vsr_sync_feats)
        np_tgts = np.asarray(tgts)
         
        return (
            np_mixtures,
             np_enrolls,
             np_codecs,
             np_visual_codecs,
            #  np_visual_enroll,
             np_visual_sync_feature,
             np_tgts

             )
    # mixtures,enrolls,codecs,visual_codecs, visual_enroll, visual_sync_feature
    

    def __len__(self):
        return len(self.minibatch)
        # return 10 


class dataset_scale_trimodal(data.Dataset):
    def __init__(
        self,
        vsr_feat_dir,
        vsr_sync_lst_train_path,
        vsr_sync_lst_path,
        visual_direc,
        aux_list,
        aux_train_list,
        mix_lst_path,
        mix_lst_train_path,
        codec_lst_path, 
        codec_lst_train_path, 
        visual_codec_lst_train_path,
        batch_size,
        partition,
        sampling_rate,
        codec_rate,
        mix_no,
        max_length,
        max_enroll_length,
        text_direc
    ):

        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.fps = 25
        self.batch_size = batch_size
        self.max_length = max_length

        self.max_enroll_length = max_enroll_length
        self.visual_direc = visual_direc
        self.vsr_feat_dir = vsr_feat_dir
        self.text_direc = text_direc
         
        if self.partition == 'train':
            self.aux_list =  open(aux_train_list).read().splitlines()[:]
            mix_lst = open(mix_lst_train_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_train_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}


            self.vsr_sync_path_dict = open(vsr_sync_lst_train_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}

        else:
            self.aux_list =  open(aux_list).read().splitlines()
            mix_lst = open(mix_lst_path).read().splitlines()[:]
            self.codec_lst = open(codec_lst_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            self.vsr_sync_path_dict = open(vsr_sync_lst_path).read().splitlines()
            self.vsr_sync_dict =  {item.split()[0]: item for item in self.vsr_sync_path_dict}


            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}
        sorted_mix_lst = mix_lst[:]
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)
        

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def pick_random_span(self, s: str, k_min: int = 3, k_max: int = 6) -> str:
            words = s.strip().split()
            n = len(words)
            if n == 0:
                return ""
            # 句子很短，直接返回全部
            if n <= k_min:
                return " ".join(words)
            # 实际可用的最大长度不超过句长
            k_max_eff = min(k_max, n)
            k_min_eff = min(k_min, k_max_eff)
            # 随机选择一个长度
            import random
            k = random.randint(k_min_eff, k_max_eff)
            # 起始下标随机，但要保证 end 不越界
            max_start = n - k
            start = random.randint(0, max_start)
            end = start + k
            return " ".join(words[start:end])
    
    def __getitem__(self, index):
        if self.partition=='train' and index==0:
            random.shuffle(self.minibatch)
        batch_lst = self.minibatch[index]
        min_length = self.max_length
 
        mixtures = []
        tgts = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        vsr_enrolls = []
        transcripts_enrolls = []
        for line in batch_lst:
            target_uttid = line.split(' ')[0]
            mixture_path = line.split(' ')[1]
            tgt_path = '/'.join(mixture_path.split('/')[:-2]) + '/s1/' + mixture_path.split('/')[-1]
            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(
                    mixture, orig_sr=sr, target_sr=self.sampling_rate
                )
            # truncate
            mixture = mixture[0 : int(min_length * self.sampling_rate)]
            if len(mixture) < int(min_length * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length * self.sampling_rate) - len(mixture)),
                )
            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)

            tgt, sr = audioread(tgt_path)
            if sr != self.sampling_rate:
                tgt = librosa.resample(tgt, orig_sr=sr, target_sr=self.sampling_rate)
            # truncate
            tgt = tgt[0 : int(min_length * self.sampling_rate)]
            if len(tgt) < int(min_length * self.sampling_rate):
                tgt = np.pad(tgt,(0, int(min_length * self.sampling_rate) - len(tgt)),)
            tgt = self.mel_proc.mel_one_np(tgt)
            tgts.append(tgt)

           
            ### read target audio codec
            codec_path = self.codec_dict[target_uttid].split(' ')[1]
            codec = np.load(codec_path)
            if codec.shape[0]< int(min_length * self.codec_rate):
                pad_length = int(min_length * self.codec_rate) - codec.shape[0]
                codec = np.pad(codec, ((0, pad_length), (0, 0)), mode='constant')
            else:
                codec = codec[:int(min_length * self.codec_rate), :]
            codecs.append(codec)

            if self.partition == 'train':

                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                if visual_codec.shape[0]< int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - visual_codec.shape[0]
                    visual_codec = np.pad(visual_codec, (0, pad_length), mode='constant')
                else:
                    visual_codec = visual_codec[:int(min_length * self.codec_rate)]
                visual_codecs.append(visual_codec)
            else:
                visual_codec = np.ones((codec.shape[0]))
                visual_codecs.append(visual_codec)
                
            #=================================read vsr ================================#
                
            if self.partition == 'train':
                vsr_sync_path = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_feat = np.load(vsr_sync_path)
                if vsr_sync_feat.shape[0]< int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    vsr_sync_feat = np.pad(vsr_sync_feat,((0, pad_length), (0, 0)),  # 只在第 0 维 pad，特征维不 pad
                        mode='constant'
                    )
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
            else:
                vsr_sync_path = self.vsr_sync_dict[target_uttid].split(' ')[1]
                vsr_sync_feat = np.load(vsr_sync_path)
                 
                if vsr_sync_feat.shape[0] < int(min_length * self.codec_rate):
                    pad_length = int(min_length * self.codec_rate) - vsr_sync_feat.shape[0]
                    vsr_sync_feat = np.pad(vsr_sync_feat,((0, pad_length), (0, 0)),  # 只在第 0 维 pad，特征维不 pad
                        mode='constant'
                    )
                else:
                    vsr_sync_feat = vsr_sync_feat[:int(min_length * self.codec_rate)]
                vsr_sync_feats.append(vsr_sync_feat)
    

            #==============================================================================

            # read enroll audio
            enroll_path = self.enroll_dict[target_uttid].split(' ')[1]
            enroll_audio, sr = audioread(enroll_path)
            if sr != self.sampling_rate:
                enroll_audio = librosa.resample(
                    enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                )
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(
                    enroll_audio,
                    (
                        0,
                        int(self.max_enroll_length * self.sampling_rate)
                        - len(enroll_audio),
                    ),
                )
            else:
                enroll_audio = enroll_audio[
                    0 : int(self.max_enroll_length * self.sampling_rate)
                ]
             
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)

            text_path = self.text_direc+ mixture_path.split('/')[-3] + '/s1/' + mixture_path.split('/')[-1].replace('.wav','.txt') 
            with open(text_path, 'r', encoding='utf-8') as file:
                full_text = file.readlines()[0].split('\n')[0]
                # 从整句里截取一个中间的 3 词 span
             # 从整句里随机截取一个长度在 [3,6] 的连续词 span
            text = full_text
            transcripts_enrolls.append(text)
            # transcripts_enrolls.append('hello, i am alex.')

 
        np_mixtures = np.asarray(mixtures)
        np_enrolls = np.asarray(enrolls)
        np_codecs = np.asarray(codecs)
        np_visual_codecs =  np.asarray(visual_codecs)
        np_visual_sync_feature =  np.asarray(vsr_sync_feats)
        np_tgts = np.asarray(tgts)

         
        return (
            np_mixtures,
             np_enrolls,
             np_codecs,
             np_visual_codecs,
            #  np_visual_enroll,
             np_visual_sync_feature,
             transcripts_enrolls,
             np_tgts,
             )
    # mixtures,enrolls,codecs,visual_codecs, visual_enroll, visual_sync_feature
    

    def __len__(self):
        return len(self.minibatch)
        # return 10 


class DatasetScaleGesture(data.Dataset):
    def __init__(
        self,
        vsr_sync_lst_train_path,
        vsr_sync_lst_path,
        aux_list,
        aux_train_list,
        mix_lst_path,
        mix_lst_train_path,
        codec_lst_path,
        codec_lst_train_path,
        visual_codec_lst_train_path,
        batch_size,
        partition,
        sampling_rate,
        codec_rate,
        mix_no,
        max_length,
        max_enroll_length,
    ):
        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_enroll_length = max_enroll_length

        # ====== 像 dataset_scale 一样构造各种 dict ======
        if self.partition == "train":
            self.aux_list = open(aux_train_list).read().splitlines()
            mix_lst = open(mix_lst_train_path).read().splitlines()

            self.codec_lst = open(codec_lst_train_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            vsr_sync_lst = open(vsr_sync_lst_train_path).read().splitlines()
            self.vsr_sync_dict = {item.split()[0]: item for item in vsr_sync_lst}

            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}
        else:
            self.aux_list = open(aux_list).read().splitlines()
            mix_lst = open(mix_lst_path).read().splitlines()

            self.codec_lst = open(codec_lst_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            vsr_sync_lst = open(vsr_sync_lst_path).read().splitlines()
            self.vsr_sync_dict = {item.split()[0]: item for item in vsr_sync_lst}

            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}

        sorted_mix_lst = mix_lst[:]

        
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        if self.partition == "train" and index == 0:
            random.shuffle(self.minibatch)

        batch_lst = self.minibatch[index]
        min_length_sec = self.max_length

        mixtures = []
        tgts = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        gestures = []

        for line in batch_lst:
            target_uttid = line.split(" ")[0]
            mixture_path = line.split(" ")[1]

            # ===== mixture & target audio → mel（沿用 dataset_scale） =====
            tgt_path = "/".join(mixture_path.split("/")[:-4]) + "/audio_clean/" + self.partition + "/" + target_uttid.split('$')[0].replace('+','/')+'.wav'

            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(mixture, orig_sr=sr, target_sr=self.sampling_rate)
            mixture = mixture[: int(min_length_sec * self.sampling_rate)]
            if len(mixture) < int(min_length_sec * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length_sec * self.sampling_rate) - len(mixture)),
                )
            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)

            tgt, sr = audioread(tgt_path)
            if sr != self.sampling_rate:
                tgt = librosa.resample(tgt, orig_sr=sr, target_sr=self.sampling_rate)
            tgt = tgt[: int(min_length_sec * self.sampling_rate)]
            if len(tgt) < int(min_length_sec * self.sampling_rate):
                tgt = np.pad(
                    tgt,
                    (0, int(min_length_sec * self.sampling_rate) - len(tgt)),
                )
            tgt = self.mel_proc.mel_one_np(tgt)
            tgts.append(tgt)

            # ===== 对应的 codec（按 dataset_scale 方式） =====
            codec_path = self.codec_dict[target_uttid].split(" ")[1]
            codec = np.load(codec_path)            # [T_codec, D_codec]
            max_codec_len = int(min_length_sec * self.codec_rate)
              
            if codec.shape[0] < max_codec_len:
                pad_len = max_codec_len - codec.shape[0]
                codec = np.pad(codec, ((0, pad_len), (0, 0)), mode="constant")
            else:
                codec = codec[:max_codec_len, :]
            codecs.append(codec)

            # 这里 visual_codec 仍然占位全 1（保持你原来的接口）
            visual_codec = np.ones((codec.shape[0]), dtype=np.float32)
            visual_codecs.append(visual_codec)

            # ===== vsr sync（沿用 dataset_scale 方式） =====
            max_gesture_len = int(min_length_sec * 15)
            vsr_sync_path = self.vsr_sync_dict[target_uttid].split(" ")[1]
            vsr_sync_feat = np.load(vsr_sync_path)    # 一般是 [T_sync, D] 或 [T_sync]
            vsr_sync_feat = vsr_sync_feat.reshape(vsr_sync_feat.shape[0], 30)
            if vsr_sync_feat.shape[0] < max_gesture_len:
                vsr_sync_feat = np.pad(vsr_sync_feat, ((0,int(max_gesture_len - vsr_sync_feat.shape[0])),(0,0)), mode = 'constant')
 
            else:
                vsr_sync_feat = vsr_sync_feat[:max_gesture_len,...]
            vsr_sync_feats.append(vsr_sync_feat)

            # ===== enroll audio（沿用 dataset_scale 方式） =====
            enroll_path = self.enroll_dict[target_uttid].split(" ")[1]
            enroll_audio, sr = audioread(enroll_path)
            if sr != self.sampling_rate:
                enroll_audio = librosa.resample(
                    enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                )
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(
                    enroll_audio,
                    (
                        0,
                        int(self.max_enroll_length * self.sampling_rate)
                        - len(enroll_audio),
                    ),
                )
            else:
                enroll_audio = enroll_audio[: int(self.max_enroll_length * self.sampling_rate)]
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)


        np_mixtures = np.asarray(mixtures, dtype=np.float32)
        np_enrolls = np.asarray(enrolls, dtype=np.float32)
        np_codecs = np.asarray(codecs, dtype=np.float32)
        np_visual_codecs = np.asarray(visual_codecs, dtype=np.float32)
        np_vsr_sync = np.asarray(vsr_sync_feats, dtype=np.float32)
        np_tgts = np.asarray(tgts, dtype=np.float32)
        return (
            np_mixtures,       # [B, T_mel, F]
            np_enrolls,        # [B, T_enroll, F]
            np_codecs,         # [B, T_codec, D_codec]
            np_visual_codecs,  # [B, T_codec]
            np_vsr_sync,       # [B, T_sync, ...]
            np_tgts,           # [B, T_mel, F]
        )


class DatasetScaleGesture_trimodal(data.Dataset):
    def __init__(
        self,
        vsr_feat_dir,
        vsr_sync_lst_train_path,
        vsr_sync_lst_path,
        visual_direc,
        aux_list,
        aux_train_list,
        mix_lst_path,
        mix_lst_train_path,
        codec_lst_path,
        codec_lst_train_path,
        visual_codec_lst_train_path,
        batch_size,
        partition,
        sampling_rate,
        codec_rate,
        mix_no,
        max_length,
        max_enroll_length,
        text_direc,
    ):
        self.minibatch = []
        self.mel_proc = MelSpec()
        self.codec_rate = codec_rate
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.C = mix_no
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_enroll_length = max_enroll_length
        self.text_direc = text_direc
        self.vsr_feat_dir = vsr_feat_dir
        self.visual_direc = visual_direc

        # ====== 像 dataset_scale 一样构造各种 dict ======
        if self.partition == "train":
            self.aux_list = open(aux_train_list).read().splitlines()
            mix_lst = open(mix_lst_train_path).read().splitlines()

            self.codec_lst = open(codec_lst_train_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            vsr_sync_lst = open(vsr_sync_lst_train_path).read().splitlines()
            self.vsr_sync_dict = {item.split()[0]: item for item in vsr_sync_lst}

            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}
        else:
            self.aux_list = open(aux_list).read().splitlines()
            mix_lst = open(mix_lst_path).read().splitlines()

            self.codec_lst = open(codec_lst_path).read().splitlines()
            self.codec_dict = {item.split()[0]: item for item in self.codec_lst}

            vsr_sync_lst = open(vsr_sync_lst_path).read().splitlines()
            self.vsr_sync_dict = {item.split()[0]: item for item in vsr_sync_lst}

            self.enroll_dict = {item.split()[0]: item for item in self.aux_list}

        sorted_mix_lst = mix_lst[:]

        
        if self.partition == "train":
            random.shuffle(sorted_mix_lst)

        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        if self.partition == "train" and index == 0:
            random.shuffle(self.minibatch)

        batch_lst = self.minibatch[index]
        min_length_sec = self.max_length

        mixtures = []
        tgts = []
        enrolls = []
        codecs = []
        visual_codecs = []
        vsr_sync_feats = []
        gestures = []
        transcripts_enrolls = []

        for line in batch_lst:
            target_uttid = line.split(" ")[0]
            mixture_path = line.split(" ")[1]

            # ===== mixture & target audio → mel（沿用 dataset_scale） =====
            tgt_path = "/".join(mixture_path.split("/")[:-4]) + "/audio_clean/" + self.partition + "/" + target_uttid.split('$')[0].replace('+','/')+'.wav'

            mixture, sr = audioread(mixture_path)
            if sr != self.sampling_rate:
                mixture = librosa.resample(mixture, orig_sr=sr, target_sr=self.sampling_rate)
            mixture = mixture[: int(min_length_sec * self.sampling_rate)]
            if len(mixture) < int(min_length_sec * self.sampling_rate):
                mixture = np.pad(
                    mixture,
                    (0, int(min_length_sec * self.sampling_rate) - len(mixture)),
                )
            mixture = self.mel_proc.mel_one_np(mixture)
            mixtures.append(mixture)

            tgt, sr = audioread(tgt_path)
            if sr != self.sampling_rate:
                tgt = librosa.resample(tgt, orig_sr=sr, target_sr=self.sampling_rate)
            tgt = tgt[: int(min_length_sec * self.sampling_rate)]
            if len(tgt) < int(min_length_sec * self.sampling_rate):
                tgt = np.pad(
                    tgt,
                    (0, int(min_length_sec * self.sampling_rate) - len(tgt)),
                )
            tgt = self.mel_proc.mel_one_np(tgt)
            tgts.append(tgt)

            # ===== 对应的 codec（按 dataset_scale 方式） =====
            codec_path = self.codec_dict[target_uttid].split(" ")[1]
            codec = np.load(codec_path)            # [T_codec, D_codec]
            max_codec_len = int(min_length_sec * self.codec_rate)
              
            if codec.shape[0] < max_codec_len:
                pad_len = max_codec_len - codec.shape[0]
                codec = np.pad(codec, ((0, pad_len), (0, 0)), mode="constant")
            else:
                codec = codec[:max_codec_len, :]
            codecs.append(codec)

            # 这里 visual_codec 仍然占位全 1（保持你原来的接口）
            visual_codec = np.ones((codec.shape[0]), dtype=np.float32)
            visual_codecs.append(visual_codec)

            # ===== vsr sync（沿用 dataset_scale 方式） =====
            max_gesture_len = int(min_length_sec * 15)
            vsr_sync_path = self.vsr_sync_dict[target_uttid].split(" ")[1]
            vsr_sync_feat = np.load(vsr_sync_path)    # 一般是 [T_sync, D] 或 [T_sync]
            vsr_sync_feat = vsr_sync_feat.reshape(vsr_sync_feat.shape[0], 30)
            if vsr_sync_feat.shape[0] < max_gesture_len:
                vsr_sync_feat = np.pad(vsr_sync_feat, ((0,int(max_gesture_len - vsr_sync_feat.shape[0])),(0,0)), mode = 'constant')
 
            else:
                vsr_sync_feat = vsr_sync_feat[:max_gesture_len,...]
            vsr_sync_feats.append(vsr_sync_feat)

            # ===== enroll audio（沿用 dataset_scale 方式） =====
            enroll_path = self.enroll_dict[target_uttid].split(" ")[1]
            enroll_audio, sr = audioread(enroll_path)
            if sr != self.sampling_rate:
                enroll_audio = librosa.resample(
                    enroll_audio, orig_sr=sr, target_sr=self.sampling_rate
                )
            if len(enroll_audio) < int(self.max_enroll_length * self.sampling_rate):
                enroll_audio = np.pad(
                    enroll_audio,
                    (
                        0,
                        int(self.max_enroll_length * self.sampling_rate)
                        - len(enroll_audio),
                    ),
                )
            else:
                enroll_audio = enroll_audio[: int(self.max_enroll_length * self.sampling_rate)]
            enroll_audio = self.mel_proc.mel_one_np(enroll_audio)
            enrolls.append(enroll_audio)

            text_path = self.text_direc+ '/'.join(vsr_sync_path.split('/')[-3:]).replace('.npy','.txt') 
            with open(text_path, 'r', encoding='utf-8') as file:
                full_text = file.readlines()[0].split('\n')[0]
                # 从整句里截取一个中间的 3 词 span
             # 从整句里随机截取一个长度在 [3,6] 的连续词 span
            text = full_text
            transcripts_enrolls.append(text)


        np_mixtures = np.asarray(mixtures, dtype=np.float32)
        np_enrolls = np.asarray(enrolls, dtype=np.float32)
        np_codecs = np.asarray(codecs, dtype=np.float32)
        np_visual_codecs = np.asarray(visual_codecs, dtype=np.float32)
        np_vsr_sync = np.asarray(vsr_sync_feats, dtype=np.float32)
        np_tgts = np.asarray(tgts, dtype=np.float32)
        return (
            np_mixtures,       # [B, T_mel, F]
            np_enrolls,        # [B, T_enroll, F]
            np_codecs,         # [B, T_codec, D_codec]
            np_visual_codecs,  # [B, T_codec]
            np_vsr_sync,       # [B, T_sync, ...]
            transcripts_enrolls,
            np_tgts,           # [B, T_mel, F]
            
        )


class DistributedSampler(data.Sampler):
    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ind = (
                torch.randperm(
                    int(len(self.dataset) / self.num_replicas), generator=g
                )
                * self.num_replicas
            )
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind + i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



# Dataset selection is explicit and driven by args.data_mode (set per mode in
# train.sh). This replaces the old manual comment/uncomment switching.
#
#   data_mode          dataset class                   scenario
#   normal             dataset                         scp-based AV (vox2/lrs3/ygd)
#   vocc_vox2          dataset_vocc_vox2               vox2 visual occlusion
#   vocc_lrs3          dataset_vocc_lrs3               lrs3 visual occlusion
#   scale              dataset_scale                   vox2-scale data
#   scale_trimodal     dataset_scale_trimodal          scale + transcript (text_direc)
#   gesture            DatasetScaleGesture             gesture features only
#   gesture_trimodal   DatasetScaleGesture_trimodal    gesture + transcript
#   switch             dataset_lrs3_switch             lrs3 switch (csv-based)
#   switch_sws_token   dataset_lrs3_switch_sws_token   switch + special token
#   memo               dataset_memo                    MEMO occlusion dataset
def get_dataloader(args, partition):
    data_mode = getattr(args, "data_mode", "normal")

    # kwargs shared by the scp-list based datasets
    scp_kwargs = dict(
        vsr_feat_dir=args.vsr_feat_dir,
        vsr_sync_lst_train_path=args.vsr_sync_lst_train_path,
        vsr_sync_lst_path=args.vsr_sync_lst_path,
        visual_direc=args.visual_dir,
        aux_list=args.aux_list,
        aux_train_list=args.aux_train_list,
        mix_lst_path=args.mix_lst_path,
        mix_lst_train_path=args.mix_lst_train_path,
        codec_lst_path=args.codec_lst_path,
        codec_lst_train_path=args.codec_lst_train_path,
        visual_codec_lst_train_path=args.visual_codec_lst_train_path,
        batch_size=args.batch_size,
    )
    length_kwargs = dict(
        max_length=args.max_length,
        max_enroll_length=args.max_enroll_length,
    )

    if data_mode == "normal":
        datasets = dataset(**scp_kwargs, partition=partition, **length_kwargs)
    elif data_mode == "vocc_vox2":
        datasets = dataset_vocc_vox2(**scp_kwargs, partition=partition, **length_kwargs)
    elif data_mode == "vocc_lrs3":
        datasets = dataset_vocc_lrs3(**scp_kwargs, partition=partition, **length_kwargs)
    elif data_mode == "scale":
        datasets = dataset_scale(**scp_kwargs, partition=partition, **length_kwargs)
    elif data_mode == "scale_trimodal":
        datasets = dataset_scale_trimodal(
            **scp_kwargs, partition=partition, sampling_rate=16000, codec_rate=25,
            mix_no=2, text_direc=args.text_direc, **length_kwargs)
    elif data_mode == "gesture":
        gesture_kwargs = {k: v for k, v in scp_kwargs.items()
                          if k not in ("vsr_feat_dir", "visual_direc")}
        datasets = DatasetScaleGesture(
            **gesture_kwargs, partition=partition, sampling_rate=16000,
            codec_rate=25, mix_no=2, **length_kwargs)
    elif data_mode == "gesture_trimodal":
        datasets = DatasetScaleGesture_trimodal(
            **scp_kwargs, partition=partition, sampling_rate=16000, codec_rate=25,
            mix_no=2, text_direc=args.text_direc, **length_kwargs)
    elif data_mode in ("switch", "switch_sws_token"):
        switch_cls = dataset_lrs3_switch if data_mode == "switch" else dataset_lrs3_switch_sws_token
        datasets = switch_cls(
            args.mix_csv_path, args.codec_dir, args.visual_dir, args.switch_audio_dir,
            args.batch_size, partition=partition, sampling_rate=16000, codec_rate=25,
            mix_no=2, **length_kwargs)
    elif data_mode == "memo":
        datasets = dataset_memo(
            getattr(args, "obj_dir", ""),
            getattr(args, "obj_mask_dir", ""),
            getattr(args, "speaker_dict", None),
            args.mix_lst_path,
            args.visual_dir,
            getattr(args, "mixture_direc", ""),
            args.batch_size,
            partition=partition,
        )
    else:
        raise ValueError(
            f"unknown data_mode '{data_mode}'. Valid: normal vocc_vox2 vocc_lrs3 "
            f"scale scale_trimodal gesture gesture_trimodal switch switch_sws_token memo"
        )

    sampler = (
        DistributedSampler(
            datasets, num_replicas=args.world_size, rank=args.local_rank
        )
        if args.distributed
        else None
    )

    generator = data.DataLoader(
        datasets,
        batch_size=1,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
    )

    return sampler, generator
