import os
import pandas as pd

def save_to_csv(data, exp_name, epoch, base_dir ='/home/jihun/dist-clipstyler/data_output2'):
    filename_map = {
        'ORI_CONT': 'ori_con_{}.csv'.format(epoch),
        'ORI_PHOTO': 'ori_photo_{}.csv'.format(epoch),
        'OURS_CONT': 'ours_{}.csv'.format(epoch)
    }
    
    if exp_name not in filename_map:
        print(f"Invalid exp_name: {exp_name}. Skipping saving.")
        return
    
    filename = os.path.join(base_dir, filename_map[exp_name])
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = df.append(data, ignore_index=True)
    # 파일이 없는 경우 새로운 DataFrame 생성
    else:
        df = pd.DataFrame([data])
    
    # 저장
    df.to_csv(filename, index=False)