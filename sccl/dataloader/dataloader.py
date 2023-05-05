"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset
import json
import numpy as np
import torch
from utils.kmeans import get_batch_token, get_mean_embeddings
from utils.optimizer import get_sbert
from sentence_transformers import SentenceTransformer

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}

class text_VirtualAugSamples(Dataset):
    def __init__(self, text):
        # assert len(train_x) == len(train_y)
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {'text': self.text[idx]}                                   

class VirtualAugSamples(Dataset):
    def __init__(self, pose, ann):
        # assert len(train_x) == len(train_y)
        self.pose = pose
        self.annotation = ann

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, idx):
        # if idx == 1478 or idx == 1477:
        #     print(idx)
        pose = self.pose.iloc[idx].poses
        # num_ann, ann = self.annotation[self.annotation.index == self.pose.iloc[idx].keyid].values[0]
        ann_emb = self.annotation[self.annotation.keyid == self.pose.iloc[idx].keyid].iloc[0].emb
        return {'pose': pose,
                # 'num_ann': num_ann,
                'annotation_emb': ann_emb} #(786,)          #TODO: (earlier) CHECK issue collating if one of the ann has more than 1 values
                                                        #RuntimeError: each element in list of batch should be of equal size


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

"""
    TODO: change to accomodate all annotations; currently using only 1st
"""
def pose_text_virtual_augmentation_loader(args):

    """
    Annotations
    """
    annotation_df = pd.read_json(os.path.join(args.datapath, args.ann_dataname+".json"))[:5]
    with open(os.path.join(args.datapath, args.ann_dataname+".json"), 'rb') as f:
        j = json.load(f)
    annotation_df['keyid'] = list(j['num_anns'].keys())[:5]                                #adding keyid as strings
    annotation_df = annotation_df[annotation_df.num_anns > 0].reset_index(drop=True)        #remove empty annotations
    annotation_df.anns = annotation_df.anns.apply(lambda x: x[:1])                          #only 1st annotation TODO
    
    #Encoding annotations to prepare dataset with pose embeddings
    print("Encoding annotations: ", len(annotation_df))
    sbert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    emb = sbert.encode([x[0] for x in annotation_df.anns.to_list()], batch_size=256, show_progress_bar=True)
    
    #make final df with encodings
    emb_df = pd.DataFrame({'emb': [x for x in emb]} )
    annotation_df['emb'] = emb_df.emb

    # annotation_df = annotation_data.set_index('keyid')
    
    """
    Poses
    """
    pose_df = pd.read_pickle(os.path.join(args.datapath, args.pose_dataname+".pkl"))

    temp_dfs = []
    for key, seq_poses in pose_df.items():
        key_df = pd.DataFrame({'poses': [pose for pose in seq_poses]} )
        key_df['keyid'] = key
        temp_dfs += [key_df]
    pose_df = pd.concat(temp_dfs, ignore_index=True)[:1000]

    empty_anns_keyid = annotation_df[annotation_df.num_anns < 1].keyid
    #adding sentence emb to df
    # if args.use_pretrain == "SBERT":
    #     sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    # annotation_df = sbert.encode([x[0] for x in annotation_df.anns.to_list()], show_progress_bar=True, batch_size=256)

    ann_mask = ~pose_df['keyid'].isin(empty_anns_keyid)       #mask: poses not with empty anns
    pose_df = pose_df[ann_mask]

    train_dataset = VirtualAugSamples(pose_df, annotation_df)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)   
    return train_loader


def test_text_virtual_augmentation_loader(args):
    annotation_df = pd.read_json(os.path.join(args.datapath, args.ann_dataname+".json"))[:50]
    annotation_df.anns = annotation_df.anns.apply(lambda x: x[:1])
    # train_text = train_data[args.text].fillna('.').values
    # train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(annotation_df).to(torch.device('cuda'))
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return train_loader

def virtual_augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label).to(torch.device('cuda'))
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    return train_loader


def unshuffle_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader

