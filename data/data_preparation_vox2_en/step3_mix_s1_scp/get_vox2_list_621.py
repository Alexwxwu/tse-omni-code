import csv
# test_csv = '/home/export/base/sc100138/sc100138/online1/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/mixture_data_list_2mix_en_test.csv'
# mix_dir = '/home/export/base/sc100138/sc100138/online1/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/test/mix/'
# s1_dir = '/home/export/base/sc100138/sc100138/online1/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/test/s1/'
# aux_dir = '/home/export/base/sc100138/sc100138/online1/voxceleb2/wav/'

# s1_scp = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_list/s1.scp'
# mix_clean_scp = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_list/mix_clean.scp'


# for normal dataset: 20000 train samples
test_csv = '/home/export/base/sc100138/sc100138/online1/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/mixture_data_list_2mix_en_train.csv'
mix_dir = '/home/export/base/sc100138/sc100138/online1/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/train/mix/'
s1_dir = '/home/export/base/sc100138/sc100138/online1/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/train/s1/'
aux_dir = '/home/export/base/sc100138/sc100138/online1/voxceleb2/wav/'
s1_scp = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list/s1.scp'
mix_clean_scp = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list/mix_clean.scp'

# for scaled dataset: 40000 train samples
test_csv = '/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/mixture_data_list_2mix_en_train.csv'
mix_dir = '/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/train/mix/'
s1_dir = '/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/train/s1/'
aux_dir = '/mnt/Corpus-Upload/Voxceleb2/origin/audio_clean/'
s1_scp = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list-scale/s1.scp'
mix_clean_scp = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list-scale/mix_clean.scp'



test_csv = '/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/mixture_data_list_2mix_en_test.csv'
mix_dir = '/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/test/mix/'
s1_dir = '/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/test/s1/'
aux_dir = '/mnt/Corpus-Upload/Voxceleb2/origin/audio_clean/'
s1_scp = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_test_list-scale/s1.scp'
mix_clean_scp = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_test_list-scale/mix_clean.scp'


#  data_direc_mp4=/mnt/Corpus-Upload/Voxceleb2/origin/video/dev/ #video folder of VoxCeleb2

with open(test_csv) as f:
    lines = f.readlines()
#!!!!!! from here to select: s1_scp or mix_clean_scp
curr_file = s1_scp
with open(curr_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    for line in lines:
        items = line.split('\n')[0].split(',')
        # import pdb;pdb.set_trace()
        if items[1] == 'dev':
             items[1] = 'train'
        if items[5] == 'dev':
             items[5] = 'train'

        mix_utt_id = items[2] + '_' + items[3].split('/')[0]+ '_' +items[3].split('/')[1]+ '_' + items[6]+ '_' + items[7].split('/')[0]+ '_' + items[7].split('/')[1]
        # import pdb;pdb.set_trace()
        # path = line.split('\n')[0].replace(',','_').replace('/','_')
        path = '_'.join(items).replace('/','_')


        mix_path = mix_dir + path +'.wav'
        s1_path = s1_dir + path  +'.wav'

        mix_line = [mix_utt_id + ' '+ mix_path]
        s1_line = [mix_utt_id + ' '+ s1_path]
        if curr_file == mix_clean_scp:
            writer.writerow(mix_line)  # 使用 writerow 而不是 writerows
        else:
            writer.writerow(s1_line)  # 使用 writerow 而不是 writerows


