import os
import re


# tse_output_dir = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/src/tse_output/wavs'
# # 输出 .scp 文件路径
# output_scp_path = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/_rej_list/wavs.scp'


# tse_output_dir = '/home/export/base/sc100135/sc100135/online1/lauraTSE_code_refact/src/tse_output_librimix_wesep_train100/wavs'
# output_scp_path = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/librimix_wesep_train100_list_rej/wavs.scp'

# tse_output_dir = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/src/tse_output_vox2_en/wavs'
# # 输出 .scp 文件路径
# output_scp_path = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_list_rej/wavs.scp'

tse_output_dir = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/src/tse_output_vox2_en_train/wavs'
# 输出 .scp 文件路径
output_scp_path = '/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list_rej/wavs.scp'
 


# 打开输出文件
with open(output_scp_path, "w") as scp_file:
    for filename in os.listdir(tse_output_dir):
        if filename.endswith(".wav") and 'vox2' in output_scp_path:
            # import pdb;pdb.set_trace()
            # base_name = '_'.join(filename.split('.wav')[0].split('train')[1:3]).replace('_0__','')

            filebase_name = filename.split('.wav')[0]
            id1_start = 12
            id1_length = 26
            id2_start = 12+34
            id2_length = 25 # 你的ID长度一样

            id1 = filebase_name[id1_start:id1_start + id1_length]
            id2 = filebase_name[id2_start:id2_start + id2_length]

            base_name = id1  + id2

            # import pdb;pdb.set_trace()
            scp_line = f"{base_name} {os.path.join(tse_output_dir, filename)}\n"
                # 写入到 .scp 文件
            # import pdb;pdb.set_trace()
            scp_file.write(scp_line)
        # 检查文件是否是 .wav 文件
        elif filename.endswith(".wav") and 'vox2' not in output_scp_path:
            base_name = filename.split('.wav')[0]
            import pdb;pdb.set_trace()
            scp_line = f"{base_name} {os.path.join(tse_output_dir, filename)}\n"
                # 写入到 .scp 文件
            # import pdb;pdb.set_trace()
            scp_file.write(scp_line)


        


# save to "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/src/tse_output/wavs/wavs.scp"

    
    


