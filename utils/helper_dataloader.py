import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, Sampler
import time
import math

# --- helper ---
def save_split(data_list, fname):
    df = pd.DataFrame(data_list, columns=["users"])
    df.to_csv('./utils' + os.sep + fname, index=False)


def random_shift(signal, cfg):
    
    base_shift = cfg['to_augment'].get('base_shift', 7)
    
    shift_amount = int(np.random.uniform(-base_shift, base_shift))
    
    shifted_signal = np.roll(signal, shift_amount)

    if shift_amount > 0:  # Right shift
        # Pad the beginning with the first non-shifted value
        shifted_signal[:shift_amount] = signal[0]
    elif shift_amount < 0:  # Left shift
        # Pad the end with the last non-shifted value
        shifted_signal[shift_amount:] = signal[-1]
        
    return shifted_signal

def random_scaling(signal, cfg):
    min_scale = cfg['to_augment'].get('min_scale', 0.8)
    max_scale = cfg['to_augment'].get('max_scale', 1.2)
    scale = np.random.uniform(min_scale, max_scale)
    return signal * scale
    
def ecg_flip(signal, cfg):
    return -signal
    
    
def min_max_norm(signal):
    min_val = signal.min()
    max_val = signal.max()
    return 2 * (signal - min_val) / (max_val - min_val) - 1


# --- Dataset Class ---
class MyDataset(Dataset):
    def __init__(self, h5_file_paths, user_ids, cfg, stage='train'):
        super().__init__()
        self.h5_file_paths = h5_file_paths
        self.user_ids = set(user_ids)
        self.cfg = cfg
        self.stage = stage
        self.scale = self.cfg.get('scale', 'all')
        
        self.augment = cfg['to_augment']['use']
        self.to_norm = True 
        self.min_qual = self.cfg['qual_thre']
        
        if isinstance(self.min_qual, float):
            self.min_qual = [self.min_qual]
        if len(self.min_qual) == 1:
            self.min_qual += self.min_qual
        
        # This master index maps an integer to the correct file and path
        self.segment_index = self._create_index()
        
        # Keep a dictionary of open file handles for each worker
        self.open_files = {}

    def _filter_file_segments(self, h5_file):
        """Filters segments from a single H5 file based on quality."""
        ppg_min_thre, ecg_min_thre = self.min_qual[0], self.min_qual[1]  
            
        kept_identifiers = []
        for user_id in h5_file.keys():
            if user_id in self.user_ids and 'vital_' not in user_id:  # remove vitaldb
            # if user_id in self.user_ids:
                for seg_id in h5_file[user_id].keys():
                    
                    ppg_q = float(seg_id.split('+++')[-2])
                    ecg_q = float(seg_id.split('+++')[-1])
                    
                    if ppg_q >=  ppg_min_thre and ecg_q >= ecg_min_thre:
                        kept_identifiers.append((user_id, seg_id, False))
                        
        return kept_identifiers

    def _create_index(self):
        """Scans all H5 files, filters them, and creates a master index."""
        index = []
        # print("Creating index from H5 files...")
        for h5file in self.h5_file_paths:
            kept_segments = self._filter_file_segments(h5file)
            
            # Add kept segments to the master index
            for user_id, seg_id, to_aug in kept_segments:
                index.append((h5file, user_id, seg_id, to_aug))
    
        total_users = [u for _, u, _, _ in index]
        
        print(f"After filtering, {len(index)} segments from {len(set(total_users))} users with min quality [ppg, ecg]: {self.min_qual}.")
        return index

    def __len__(self):
        return len(self.segment_index)

    def __getitem__(self, idx):
        h5_file, user_id, seg_id, to_aug = self.segment_index[idx]
        
    
        data_array = h5_file[f'{user_id}/{seg_id}'][:]
        assert data_array.shape[0] == self.cfg['seg_len'] * self.cfg['sampling_freq']
        
        demo_dict = {
            'user_id': user_id,
        }

        signal_dict = self._ppg_ecg_iter(data_array)
            
        for signal_type in signal_dict:
            signal = signal_dict[signal_type]

            if self.stage != 'train' and (self.augment or to_aug):
                if random.random() > 0.5:
                    signal_dict[signal_type] = random_scaling(signal, self.cfg)
                else:
                    signal_dict[signal_type] = random_shift(signal, self.cfg)
                    
                if signal_type == 'ecg' and random.random() > 0.5:
                    signal_dict[signal_type] = ecg_flip(signal_dict[signal_type], self.cfg)
            
            if self.to_norm:
                signal_dict[signal_type] = min_max_norm(signal_dict[signal_type])
                
            signal_dict[signal_type] = torch.from_numpy(signal_dict[signal_type]).float()

        return {
            **signal_dict,
            **demo_dict,
        }
        
    
    def _ppg_ecg_iter(self, data_array):
        return {'ppg': data_array[:, 1], 'ecg': data_array[:, 2]}


    
# --- PyTorch Lightning DataModule ---
class MyDataModule(pl.LightningDataModule):
    def __init__(self, h5_file, cfg):
        super().__init__()
        self.h5_file_paths = h5_file
        self.cfg = cfg
        self.batch_size = cfg['train']['batch_size']
        self.num_workers = cfg['data']['num_workers']

        self.val_split = cfg['data']['val_split']
        self.test_split = cfg['data']['test_split']
        self.random_seed = cfg['random_seed']
        
        self.dataset_names = '++'.join(cfg['dataset'] + [cfg['data']['type']] + [str(cfg['random_seed'])])
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        random.seed(self.random_seed)

        
    def setup(self, stage=None):
        before = time.time()
        # Scan all files to get a complete list of users for splitting
        all_user_ids = []
        for h5_file in self.h5_file_paths:
            all_user_ids += list(h5_file.keys())
  
        all_user_ids = sorted(list(set(all_user_ids)))
        print(f"Found {len(all_user_ids)} unique users across all files before filtering.")

        all_user_ids = [u for u in all_user_ids if 'vital_' not in u] # remove vitaldb
        
        # 2. Split user IDs into train, validation, and test sets
        train_val_users, test_users = train_test_split(
            all_user_ids,
            test_size=self.test_split,
            random_state=self.random_seed
        )
        
        # Adjust validation split relative to the remaining data
        train_users, val_users = train_test_split(
            train_val_users,
            test_size=self.val_split,
            random_state=self.random_seed
        )
        
        assert len(set(train_users).intersection(val_users)) == 0
        assert len(set(train_users).intersection(test_users)) == 0
        assert len(set(val_users).intersection(test_users)) == 0
        
        save_split(train_users, self.dataset_names + '+++' + 'train.csv')
        save_split(val_users ,  self.dataset_names + '+++' + 'val.csv')
        save_split(test_users,  self.dataset_names + '+++' + 'test.csv')
        
        print(f'train users: {len(train_users)}')
        print(f'val   users: {len(val_users)}')
        print(f'test  users: {len(test_users)}')

        # 3. Create the datasets for each split
        if stage == 'fit' or stage is None:
            self.train_dataset = MyDataset(self.h5_file_paths, train_users, self.cfg, stage='train')
            self.val_dataset = MyDataset(self.h5_file_paths, val_users, self.cfg, stage='val')
        if stage == 'test' or stage is None:
            self.test_dataset = MyDataset(self.h5_file_paths, test_users, self.cfg, stage='test')

        print(f'[INFO] datamodule is ready with {time.time() - before:.3f} seconds.')

    def _get_loader(self, dataset, stage):

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if stage == 'train' else False,
            num_workers=self.num_workers,
            pin_memory=True
        )
            
    def train_dataloader(self):
        return self._get_loader(self.train_dataset, 'train')

    def val_dataloader(self):
        return self._get_loader(self.val_dataset, 'val')

    def test_dataloader(self):
        return self._get_loader(self.test_dataset, 'test')
       




