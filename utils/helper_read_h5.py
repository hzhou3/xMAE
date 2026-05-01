import h5py
import boto3
import io, os
import numpy as np
import pickle
import time


def get_dataset_path(cfg):
    # --- Data Loading ---
    dataset_paths = []
    
    if isinstance(cfg["dataset"], str):
        cfg["dataset"] = [cfg["dataset"]]
    bucket = cfg['from_s3']['bucket']
    s3_key = cfg['from_s3']['key'].replace('{dataset}', f'{cfg["dataset"]}')
        
    for dataset in cfg["dataset"]:
        bucket = cfg['from_s3']['bucket']
        s3_key = cfg['from_s3']['key'].replace('{dataset}', f'{dataset}')
        
        if dataset == 'pulsedb':
            for shard in range(27):  
                temp_key = s3_key + f'r_normalized_seg-{cfg["seg_len"]}s/shard_{shard}_seg-{cfg["seg_len"]}s_{cfg["sampling_freq"]}hz.h5'
                print(f'loading {temp_key}')
                dataset_paths.append((bucket, temp_key))

        else:
            continue
        
    if not len(dataset_paths):
        print('No such dataset')
        exit()

    return dataset_paths
        

def read_h5_from_s3(cfg):
    before = time.time()
    h5s = []
    print(f"Loading dataset: `{cfg['dataset']}`...")
    for bucket_name, s3_key in get_dataset_path(cfg):
        temp = _read_one_h5_from_local(bucket_name, s3_key)

        if temp is None:
            break
        h5s.append(temp)

    print(f"Loading dataset from s3 takes {time.time()-before:.3f} seconds!")

    return h5s
    


def _read_one_h5_from_local(bucket_name, s3_key):
              
    path = '../' + os.path.join(bucket_name, s3_key)
        
    # 1. Check if the local file exists
    if not os.path.exists(path):
        print('Local copy not found, reading from s3...')
        return _read_one_h5_from_s3(bucket_name, s3_key)

    # 2. Open the file directly from disk
    h5_file = h5py.File(path, 'r')
    return h5_file

    
def _read_one_h5_from_s3(bucket_name, s3_key):
    

    try:
        # 1. Create an S3 client
        s3_client = boto3.client('s3')

        # 2. Get the object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        # 3. Read the object's content into an in-memory buffer
        # The 'Body' of the response is a streaming object
        file_content = response['Body'].read()
        buffer = io.BytesIO(file_content)
        
        # 4. Open the buffer with h5py as if it were a file
        # print(f"Successfully loaded {s3_key} from bucket {bucket_name} into memory.")
        h5_file = h5py.File(buffer, 'r')
        
        return h5_file

    except Exception as e:
        print(f"Error reading from S3: {e}")
        exit()



if __name__ == "__main__":
    pass

    