from mat73 import loadmat 
import numpy as np
import glob
import pandas as pd
import neurokit2 as nk
import io
import h5py
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm 
import boto3

meta_info_keys = ['Age', 'BMI', 'CaseID', 'Gender', 'Height', 'Weight', 'IncludeFlag', 'SegDBP', 'SegSBP', 'SegmentID', 'SubjectID', 'WinID', 'WinSeqID']
bp_keys = ['SegDBP', 'SegSBP' ]
ignore_keys = ['ABP_SPeaks', 'ABP_Turns', 'ECG_RPeaks', 'PPG_SPeaks', 'PPG_Turns']
ecg_keys = ['ECG_F', 'ECG_Raw', 'ECG_Record', 'ECG_Record_F']
ppg_keys = ['PPG_F', 'PPG_Raw', 'PPG_Record', 'PPG_Record_F'] 
time_keys = ['T']

SP = "+++" 
FS = 125
NEW_FS = 100
S3_BUCKET = "YOUR_BUCKET"
DOWNLOADED_DATA = './raw_dataset/pulsedb/raw_data/PulseDB_*/*.mat'

def get_data(data, key, length, dataset):

    try:
        shapes = np.array(data[key]).shape
        l = 1
        for s in shapes:
            l *= s
    except:
        l = length

    assert l == length
    
    try:
        d = np.array(data[key]).reshape(length, )
    except:
        d = np.array([-1] * length) 

    if key == 'Gender':
        d = d == 'M'


    assert d.shape[0] == length

    if key == 'SubjectID':
        return [dataset+"_"+i for i in list(d)]
    
    return list(d)
   
def min_max_norm(df):
    col_to_norm = df.columns[1:3]

    signal = df[col_to_norm]
    
    min_val = signal.min()
    max_val = signal.max()

    df[col_to_norm] = 2 * (signal - min_val) / (max_val - min_val) - 1

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].astype('float16')

    return df

    
def get_demo_data(data, length, dataset):

    return {key: get_data(data, key, length, dataset) for key in meta_info_keys}


def process_signal(ppg, ecg):
        
    # 3. Clean the ECG signal using NeuroKit2
    cleaned_ecg = nk.ecg_clean(ecg, sampling_rate=FS)
    cleaned_ppg = nk.ppg_clean(ppg, sampling_rate=FS)
        
    e_qual = np.percentile(nk.ecg_quality(cleaned_ecg, sampling_rate=FS, method='templatematch'), 15)
    p_qual = np.percentile(nk.ppg_quality(cleaned_ppg, sampling_rate=FS, method='templatematch'), 15)
    
    down_ecg = nk.signal_resample(
        cleaned_ecg,
        sampling_rate=FS,
        desired_sampling_rate=NEW_FS,
    )

    down_ppg = nk.signal_resample(
        cleaned_ppg,
        sampling_rate=FS,
        desired_sampling_rate=NEW_FS,
    )

    ts = np.linspace(0, 10, len(down_ppg), endpoint=False)
    
    df = pd.DataFrame({
        'ts': ts,
        'ppg': down_ppg,
        'ecg': down_ecg,
    })

    return e_qual, p_qual, min_max_norm(df)





def read_one_mat(fpath):

    dataset = fpath.split('/')[-2].replace('PulseDB_', '').lower()
    
    data = loadmat(fpath)
    
    data = data['Subj_Wins']

    ecg = np.array(data['ECG_Raw'])
    ppg = np.array(data['PPG_Raw'])
    ts = np.array(data['T'])


    ecg = ecg.reshape(-1, ecg.shape[-1])
    ppg = ppg.reshape(-1, ppg.shape[-1])
    ts = ts.reshape(-1, ts.shape[-1])

    assert ecg.shape == ppg.shape
    assert ecg.shape == ts.shape

    demo_dict = get_demo_data(data, ecg.shape[0], dataset)

    ret = []
    meta = []

    for i in range(ecg.shape[0]):
        try:
            e_qual, p_qual, signal_df = process_signal(ppg[i], ecg[i])
    
            second_id = str(i).zfill(4) + SP +str(demo_dict['SegmentID'][i]) + SP + str(demo_dict['WinID'][i]) + SP + str(demo_dict['WinSeqID'][i]) + SP + f"{p_qual:.3f}" + SP + f"{e_qual:.3f}"
            # second_id = str(demo_dict['SegmentID'][i]) + SP + f"{p_qual:.3f}" + SP + f"{e_qual:.3f}"
            first_id = str(demo_dict['SubjectID'][i])
    
            ret.append((first_id, second_id, signal_df.to_numpy() ))
            meta.append([demo_dict[ks][i] for ks in meta_info_keys])
                
        except:
            pass


    return (ret, meta)
    


def save_h5_to_s3(all_data, shard):

    print("Creating HDF5 file in memory...")
    
    # Use io.BytesIO() to create an in-memory binary buffer
    with io.BytesIO() as h5_buffer:
        # Use h5py to write to the in-memory buffer
        with h5py.File(h5_buffer, 'w') as hf:
            # Iterate through your data and save it
            for user_id, seg_id, df in all_data:
                path = f'{user_id}/{seg_id}'
                hf.create_dataset(path, data=df)
               
        
        # Get the binary content from the buffer after it's been written
        buffer_to_upload = h5_buffer.getvalue()

    print(f"{shard} -- HDF5 file created successfully in memory. Uploading to S3...")

    # Upload the in-memory buffer to S3
    try:
        s3_client = boto3.client('s3')
        key = f'pulsedb/r_normalized_seg-10s/shard_{shard}_seg-10s_100hz.h5'
        s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=buffer_to_upload)
        print(f"Successfully uploaded {key}.")
    except Exception as e:
        print(f"Error uploading to S3: {e}")



def parallel_process(all_files):
    n_processes = os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(tqdm(executor.map(read_one_mat, all_files), total=len(all_files)))

    signals = []
    combined_meta = []
    for ret, meta in results: 
        if len(ret) == 0:
            continue
        signals += ret
        combined_meta += meta

    print('creating meta df...')
    meta_df = pd.DataFrame(data=combined_meta, columns=meta_info_keys)
    print('uploading meta df to s3')
    try:
        
        s3_path = f"s3://{S3_BUCKET}/pulsedb/meta.csv"
        meta_df.to_csv(s3_path, index=False)
        
    except Exception as e:
        print(f"Error uploading meta df to S3: {e}") 


    del meta_df

    file_len = 200000 # save periodically to prevent OOM
    for shard, start_idx in enumerate(range(0, len(signals), file_len)):
        save_h5_to_s3(signals[start_idx:start_idx+file_len], shard)

    print('DONE')



if __name__ == '__main__':
        
    
    all_files = glob.glob(DOWNLOADED_DATA)
    
    print(len(all_files))

    parallel_process(all_files)
    