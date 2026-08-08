import os
import numpy as np
import argparse
import csv
import tqdm
import librosa
import scipy.io.wavfile as wavfile
# import ffmpeg
# from moviepy.editor import VideoFileClip



EPS = np.finfo(float).eps
np.random.seed(0)


def extract_wav_from_mp4(line):
    # import pdb;pdb.set_trace()
    args.data_direc = '/mnt/Corpus-Upload/Voxceleb2/origin/video/dev/'
    args.audio_data_direc = '/mnt/Corpus-Upload/Voxceleb2/origin/audio_clean/dev/'

    # Extract .wav file from mp4
    video_from_path = (
        args.data_direc + line[0] + "/" + line[1] + "/" + line[2] 
    )
    audio_save_path = (
        args.audio_data_direc + line[0] + "/" + line[1] + "/" + line[2].replace('mp4', 'wav')
    )
    if not os.path.exists(video_from_path):
        return
    if not os.path.exists(audio_save_path.rsplit("/", 1)[0]):
        os.makedirs(audio_save_path.rsplit("/", 1)[0])
    if not os.path.exists(audio_save_path):
        os.system(
            "ffmpeg -i %s %s" % (video_from_path, audio_save_path)
        )  # if audio not exist, then extract audio from video
        # ffmpeg.input(video_from_path).output(audio_save_path).run()
        
        # clip = VideoFileClip(video_from_path)
        # clip.audio.write_audiofile(audio_save_path)

        # import subprocess
        # # 构造命令列表
        # cmd = ['ffmpeg', '-i', video_from_path, audio_save_path]
        # subprocess.run(cmd, check=True)


    sr, audio = wavfile.read(audio_save_path)
    assert sr == args.sampling_rate, "sampling_rate mismatch"
    sample_length = audio.shape[0]
    return sample_length  # In seconds


def get_test_list(args):
        audio_info_list = args.audio_info_list  
        audio_list = args.audio_list  
        f1 = open(audio_info_list, "w")
        w1 = csv.writer(f1)
        test_list = []
        # import pdb;pdb.set_trace()
        with open(audio_list) as f:
            lines = f.readlines()
         
        for line in lines:
            # import pdb;pdb.set_trace()
            line = line.split(' ')[:2]
            if line[1] == 'en':
                ln = line[0].split('/')[1:]
                ln[-1] = ln[-1] + '.mp4'
                # import pdb;pdb.set_trace()
            else:
                continue
            sample_length = extract_wav_from_mp4(ln)
            if sample_length == None:
                continue
            if sample_length < args.min_length * args.sampling_rate :
                continue
            # import pdb;pdb.set_trace()
            ln[-1] = ln[-1].split('.')[0]
            # import pdb;pdb.set_trace()
            ln += [sample_length / args.sampling_rate]
            ln = ['train'] + ln
            # import pdb;pdb.set_trace()
            w1.writerow(ln)
            
            # test_list.append(ln)

        # return test_list


def main(args):
    # get_test_list(args)
    import pdb;pdb.set_trace()
    # read the datalist and separate into train, val and test set
    # train_list = []
    # val_list = []
    # test_list = []
    # tmp_list = []
    info = args.audio_info_list  
    with open(info) as f:
        lines = f.readlines()


    print("Gathering file names")


    # Sort the speakers with the number of utterances in pretrain set
    speakers = {}
    for ln in lines:
        ln = ln.split(',')
        ID = ln[1]
        if ID not in speakers:
            speakers[ID] = 1
        else:
            speakers[ID] += 1
    sort_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)
    import pdb;pdb.set_trace()
    # top_selected_speakers = [spk for (spk, count) in sort_speakers[:800]]
    top_selected_speakers = [spk for (spk, count) in sort_speakers[:118]]


    # Create mixture list
    print("Creating mixture list")
    f = open(args.mixture_data_list, "w")
    w = csv.writer(f)
    # import pdb;pdb.set_trace()

    top_lines = []
    for line in lines:
        # import pdb;pdb.set_trace()
        line = line.split(',')
        if line[1] not in top_selected_speakers:
            continue
        else:
            # import pdb;pdb.set_trace()
            line = [line[0], line[1], line[2]+'/'+line[3], line[4].split('\n')[0]]
            top_lines.append(line)
    # import pdb;pdb.set_trace()
    # args.test_samples=2000
    data_list = top_lines
    data = "test"
    # data = 'train'
    # data = 'val'
    length = args.test_samples
    count_list = []

    import pdb;pdb.set_trace()
    for ln in data_list:
        if not ln[1] in count_list:
            count_list.append(ln[1])
    print(
        "In %s list: %s speakers, %s utterances"
        % (data, len(count_list), len(data_list))
    )

    for _ in range(length):
        mixtures = [data]
        shortest = 200
        cache = []
        while len(cache) < args.C:
            # import pdb;pdb.set_trace()
            idx = np.random.randint(0, len(data_list))
            if data_list[idx][1] in cache:
                continue
            cache.append(data_list[idx][1])
            mixtures = mixtures + list(data_list[idx])
            if float(mixtures[-1]) < shortest:
                shortest = float(mixtures[-1])
            del mixtures[-1]
            if len(cache) == 1:
                db_ratio = 0
            else:
                db_ratio = np.random.uniform(-args.mix_db, args.mix_db)
            mixtures.append(db_ratio)
        mixtures.append(shortest)
        # import pdb;pdb.set_trace()
        w.writerow(mixtures)

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxceleb2 dataset")
    parser.add_argument("--data_direc", type=str)
    parser.add_argument("--C", type=int)
    parser.add_argument("--mix_db", type=float)
    # parser.add_argument("--train_samples", type=int)
    # parser.add_argument("--val_samples", type=int)
    parser.add_argument("--test_samples", type=int)
    # parser.add_argument("--audio_data_direc", type=str)
    parser.add_argument("--min_length", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--sampling_rate", type=int)
    parser.add_argument("--mixture_data_list", type=str)
    parser.add_argument("--audio_info_list", type=str)
    parser.add_argument("--audio_list", type=str)
    args = parser.parse_args()

    main(args)
