import csv

#find . -type f | wc -l

# # 输入和输出csv文件路径
# input_csv = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/train/all.scp'
# output_csv = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/all_vsr_feat.scp'
# visual_codec_dir = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/dev/'
# # 定义目标字符串长度（即用字符串个数划分）

# 输入和输出csv文件路径
input_csv = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/test/all.scp'
output_csv = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/test_all_vsr_feat.scp'
visual_codec_dir = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/dev/'


input_csv = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/train_scale/all.scp'
output_csv = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/train_vsr_continous_feat_scale/all_vsr_feat_scale.scp'
visual_codec_dir = '/mnt/users/hccl.local/wwu/Voxceleb2/muse/lip/dev/'


input_csv = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/test_scale/all.scp'
output_csv = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/train_vsr_continous_feat_scale/test_all_vsr_feat_scale.scp'
visual_codec_dir = '/mnt/users/hccl.local/wwu/Voxceleb2/muse/lip/test/'

# 定义目标字符串长度（即用字符串个数划分）
 

with open(input_csv, 'r', encoding='utf-8') as f_in, \
     open(output_csv, 'w', encoding='utf-8', newline='') as f_out:

    reader = csv.reader(f_in, delimiter=';')
    writer = csv.writer(f_out, delimiter=' ')

    for row in reader:
        # import pdb;pdb.set_trace()
        target_id = row[0].split(' ')[0]
        target_path = row[0].split(' ')[1]
        # 提取路径中的目标ID部分（路径里的第一个合适字符串）
        # 这里用`target_path`字符串，截取长度后作为“目标ID”
        # 如果路径中有编号或目标ID在第一个文件夹名，用类似的逻辑
        target_first_id =  target_id[:7] + '/' + target_id[8:8+11] + '/'+ target_id[20:20+5]
        target_path_new = visual_codec_dir + target_first_id +'.npy'
        # import pdb;pdb.set_trace()
        writer.writerow([target_id, target_path_new])