import os
import csv

# 文件路径
# test_csv = '/home/export/base/sc100138/sc100138/online1/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/mixture_data_list_2mix_en_test.csv'
# all_list = '/home/export/base/sc100138/sc100138/online1/vsr-low-tse-dataset/vsr-low/file.list'
# video_dir = '/home/export/base/sc100138/sc100138/online1/voxceleb2/mp4/'
# audio_dir = '/home/export/base/sc100138/sc100138/online1/voxceleb2/wav/'
# vox2_enroll_scp = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_list/aux_s1.scp'

test_csv = '/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/mixture_data_list_2mix_en_train.csv'
all_list = '/mnt/users/hccl.local/wwu/vsr-low/file.list'
video_dir = '/mnt/Corpus-Upload/Voxceleb2/origin/video/'
audio_dir = '/mnt/Corpus-Upload/Voxceleb2/origin/audio_clean/'
vox2_enroll_scp = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list-scale/aux_s1.scp'



test_csv = '/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/mixture_data_list_2mix_en_test.csv'
all_list = '/mnt/users/hccl.local/wwu/vsr-low/file.list'
video_dir = '/mnt/Corpus-Upload/Voxceleb2/origin/video/'
audio_dir = '/mnt/Corpus-Upload/Voxceleb2/origin/audio_clean/'
vox2_enroll_scp = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_test_list-scale/aux_s1.scp'
# 读取测试 CSV 文件
with open(test_csv) as f:
    test_lines = f.readlines()

# 读取所有列表文件
with open(all_list) as f1:
    all_lines = f1.readlines()

# 打开输出 CSV 文件（只打开一次）
with open(vox2_enroll_scp, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # 遍历测试 CSV 文件的每一行
    count = 0
    for line in test_lines:
        # if count < 16495:
        #     count+=1
        #     # import pdb;pdb.set_trace()
        #     continue
        # import pdb;pdb.set_trace()
         
        items = line.split('\n')[0].split(',')
        aux_id = items[2]

        # 在所有列表文件中查找匹配的条目
        for aux_line in all_lines:
            aux_item = aux_line.split('\n')[0].split('/')
            # import pdb;pdb.set_trace()
            if aux_item[2] == aux_id:
                # 根据 dev 或 test 设置路径
                if aux_item[1] == 'dev':
                    aux_video_path = os.path.join(video_dir, 'dev', *aux_item[2:]) + '.mp4'
                    audio_save_path = os.path.join(audio_dir, 'dev', *aux_item[2:]) + '.wav'
                else:
                    aux_video_path = os.path.join(video_dir, 'test', *aux_item[2:]) + '.mp4'
                    audio_save_path = os.path.join(audio_dir, 'test', *aux_item[2:]) + '.wav'

                # 创建目录（如果不存在）
                # import pdb;pdb.set_trace()
                os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)

                # 如果音频文件已经存在，跳过转换
                if not os.path.isfile(audio_save_path):
                    # 使用 ffmpeg 转换视频为音频
                    os.system(f"ffmpeg -y -i {aux_video_path} {audio_save_path}")
                # else:
                #     print(f"file exist, skip: {audio_save_path}")

                # # 创建目录（如果不存在）
                # os.makedirs(os.path.dirname(audio_save_path), exist_ok=True)

                # # 使用 ffmpeg 转换视频为音频
                # os.system(f"ffmpeg -y -i {aux_video_path} {audio_save_path}")

                # 构建 CSV 行数据
                # import pdb;pdb.set_trace()
                aux_id_csv = [ '_'.join(items[2:4] + items[6:8]).replace('/','_') + ' ' + audio_save_path]
                # import pdb;pdb.set_trace()
                # 写入 CSV 文件
                writer.writerow(aux_id_csv)
                
                break