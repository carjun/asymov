"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

class VirtualAugSamples(Dataset):
    def __init__(self, pose, ann):
        # assert len(train_x) == len(train_y)
        self.pose = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}

    
class ExplitAugSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'augmentation_1': self.train_x1[idx], 'augmentation_2': self.train_x2[idx], 'label': self.train_y[idx]}
       

def explict_augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_text1 = train_data[args.augmentation_1].fillna('.').values
    train_text2 = train_data[args.augmentation_2].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = ExplitAugSamples(train_text, train_text1, train_text2, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


def pose_text_virtual_augmentation_loader(args):
    annotation_data = pd.read_json(os.path.join(args.datapath, args.ann_dataname+".json"))
    with open('data/anns.json', 'rb') as f:
        j = json.load(f)
    pose_data = pd.read_pickle(os.path.join(args.datapath, args.pose_dataname+".pkl"))

    annotation_data['keyid'] = j['num_anns'].keys()
    annotation_df = annotation_data.set_index('keyid')

    temp_dfs = []
    for key, seq_poses in pose_data.items():
        key_df = pd.DataFrame({'poses': [pose for pose in seq_poses]} )
        key_df['keyid'] = key
        temp_dfs += [key_df]
    pose_df = pd.concat(temp_dfs, ignore_index=True)

    # train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    # train_text = train_data[args.text].fillna('.').values
    # train_label = train_data[args.label].astype(int).values

    # train_dataset = VirtualAugSamples(train_text, train_label)
    train_dataset = VirtualAugSamples(pose_df, annotation_df)
    
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    return train_loader

def virtual_augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    return train_loader


def unshuffle_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader

