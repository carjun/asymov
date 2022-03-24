import argparse
import os
import pprint
import shutil
import time
import sys
import yaml
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from pathlib import Path

import pdb
import json

from src.data.dataset.loader import KITDataset
from src import algo
from src.data.dataset.cluster_misc import lexicon#, get_names, genre_list

from plb.models.self_supervised import TAN
# from plb.models.self_supervised.tan import TANEvalDataTransform, TANTrainDataTransform
# from plb.datamodules import SeqDataModule
from plb.datamodules.data_transform import body_center, euler_rodrigues_rotation

KEYPOINT_NAME = ['root','BP','BT','BLN','BUN','LS','LE','LW','RS','RE','RW',
                'LH','LK','LA','LMrot','LF','RH','RK','RA','RMrot','RF']

import pytorch_lightning as pl
pl.utilities.seed.seed_everything(0)

def plain_distance(a, b):
    return np.linalg.norm(a - b, ord=2)

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--data_dir',
                        help='path to aistplusplus data directory from repo root',
                        type=str)
    parser.add_argument('--data_name',
                        help='which version of the dataset, subset or not',
                        default=1,
                        type=str)

    parser.add_argument('--seed',
                        help='seed for this run',
                        default=1,
                        type=int)

    parser.add_argument('--log_ver',
                        help='version in kitml_logs',
                        default=1,
                        type=int)

    parser.add_argument('--use_raw',
                        help='whether to use raw skeleton for clustering',
                        default=0,
                        type=int)

    args, _ = parser.parse_known_args()
    print(f'SEED: {args.seed}')
    pl.utilities.seed.seed_everything(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.cfg, 'r') as stream:
        ldd = yaml.safe_load(stream)

    # if args.log_ver:
    ldd["CLUSTER"]["VERSION"] = str(args.log_ver)
    ldd["CLUSTER"]["USE_RAW"] = args.use_raw
    ldd["PRETRAIN"]["DATA"]["DATA_NAME"] = args.data_name
    if args.data_dir:
        ldd["PRETRAIN"]["DATA"]["DATA_DIR"] = args.data_dir
    pprint.pprint(ldd)
    return ldd


def create_log_viz_dirs(args):
    dirname = Path(args['LOG_DIR'])
    dirname.mkdir(parents=True, exist_ok=True)
    timed = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args['LOG_DIR'], f"config_used_{timed}.yaml"), "w") as stream:
        yaml.dump(args, stream, default_flow_style=False)
    video_dir = os.path.join(args['LOG_DIR'], "saved_videos")
    Path(video_dir).mkdir(parents=True, exist_ok=True)


def get_model(args):
    '''Identify checkpoint to use, create log files, and  return model'''
    print('Using TAN model\'s features for clustering')
    load_name = args["CLUSTER"]["CKPT"] if args["CLUSTER"]["CKPT"] != -1 else args["NAME"]
    with open(os.path.join(args['LOG_DIR'], f"val_cluster_zrsc_scores.txt"), "a") as f:
        f.write(f"EXP: {load_name}\n")
    cfg = None
    for fn in os.listdir(os.path.join("./kit_logs", load_name)):
        if fn.endswith(".yaml"):
            cfg = fn
    with open(os.path.join("./kit_logs", load_name, cfg), 'r') as stream:
        old_args = yaml.safe_load(stream)
    cpt_name = os.listdir(os.path.join("./kit_logs", load_name, "default/version_" + args["CLUSTER"]["VERSION"] + "/checkpoints"))[0]
    print(f"We are using checkpoint: {cpt_name}")
    model = eval(old_args["PRETRAIN"]["ALGO"]).load_from_checkpoint(os.path.join("./kit_logs", load_name, "default/version_" + args["CLUSTER"]["VERSION"] + "/checkpoints", cpt_name))
    return model


def get_feats(args, ldd, model=None):
    ''''''
    feats = None
    if int(args["CLUSTER"]["USE_RAW"]) == 0:
        ldd1 = torch.Tensor(ldd).flatten(1, -1) #/ 100  # [T, 63]
        ttl = ldd1.shape[0]
        ct = body_center(ldd1[0])
        ldd1 -= ct.repeat(args['NUM_JOINTS']).unsqueeze(0)
        res1 = model(ldd1.unsqueeze(0).to(args['DEVICE']),
                     torch.tensor([ttl]).to(args['DEVICE']))
        forward_feat = res1[:, 0]  # [T1, f]
        forward_feat /= torch.linalg.norm(forward_feat, dim=-1, keepdim=True, ord=2)
        feats = forward_feat
    else:
        # to get results for using raw skeleton, swap with
        ldd1 = torch.Tensor(ldd).flatten(1, -1) / 100  # [T, 51]
        ttl = ldd1.shape[0]
        ct = body_center(ldd1[0])
        ldd1 -= ct.repeat(args['NUM_JOINTS']).unsqueeze(0)
        feats = ldd1
    return feats


def main():

    args = parse_args()

    # KIT Dataset configs
    args['NUM_JOINTS'] = 21
    args['LOG_DIR'] = os.path.join("./kit_logs", args["NAME"])

    # Create log, viz. dirs
    create_log_viz_dirs(args)

    # Load KIT Dataset from stored pkl file (e.g., xyz_data.pkl)
    official_loader = KITDataset(args["PRETRAIN"]["DATA"]["DATA_DIR"], args["PRETRAIN"]["DATA"]["DATA_NAME"])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['DEVICE'] = device

    # Load model only if we are using TAN Featuers ("USE_RAW" == 0)
    model = None
    if int(args["CLUSTER"]["USE_RAW"]) == 0:
        model = get_model(args)
        torch.set_grad_enabled(False)
        model.eval()
        model = model.to(args['DEVICE'])

    # Get data
    tr_kpt_container = []
    tr_len_container = []
    tr_feat_container = []
    tr_name_container = []
    val_kpt_container = []
    val_len_container = []
    val_feat_container = []
    val_name_container = []

    with open(os.path.join(args["PRETRAIN"]["DATA"]["DATA_DIR"], args["PRETRAIN"]["DATA"]["DATA_NAME"] + '_data_split.json'), 'r') as handle:
        data_split = json.load(handle)
    tr_df, val_df = data_split['train'], data_split['val'] + data_split['test'] + official_loader.filter_file

    print(f"Training samples = {len(tr_df)}\nValidation samples = {len(val_df)}")

    for reference_name in tqdm(tr_df, desc='Loading training set features'):


        try:
            ldd = official_loader.load_keypoint3d(reference_name)

            # FIXME: Temp. debug hack -- truncate to T=5000
            ldd = ldd[:5000, :, :]

            # print(reference_name, ldd.shape[0])
            tr_kpt_container.append(ldd)
            tr_len_container.append(ldd.shape[0])
            feats = get_feats(args, ldd, model)
            tr_feat_container.append(feats.detach().cpu().numpy())
            tr_name_container.append(reference_name)
        except:
            print(f'ERROR w/ seq. {reference_name}. In except: block')

    for reference_name in tqdm(val_df, desc='Loading validation set features'):
        try:
            ldd = official_loader.load_keypoint3d(reference_name)

            # FIXME: Temp. debug hack -- truncate to T=5000
            ldd = ldd[:5000, :, :]

            val_kpt_container.append(ldd)
            val_len_container.append(ldd.shape[0])
            feats = get_feats(args, ldd, model)
            val_feat_container.append(feats.detach().cpu().numpy())
            val_name_container.append(reference_name)
        except:
            print(f'ERROR w/ seq. {reference_name}. In except: block')

    print('Done loading TAN/raw position features.')
    pdb.set_trace()

    tr_where_to_cut = [0, ] + list(np.cumsum(np.array(tr_len_container)))
    tr_stacked = np.vstack(tr_feat_container)
    val_where_to_cut = [0, ] + list(np.cumsum(np.array(val_len_container)))
    val_stacked = np.vstack(val_feat_container)

    for K in range(args["CLUSTER"]["K_MIN"], args["CLUSTER"]["K_MAX"], 10):
        # get cluster centers
        argument_dict = {"distance": plain_distance, "TYPE": "vanilla", "K": K, "TOL": 1e-4}
        if not os.path.exists(os.path.join(args['LOG_DIR'], f"advanced_centers_{K}.npy")):
            print('Finding cluster centers')
            c = getattr(algo, args["CLUSTER"]["TYPE"])(tr_stacked, times=args["CLUSTER"]["TIMES"], argument_dict=argument_dict)
            np.save(os.path.join(args['LOG_DIR'], f"advanced_centers_{K}.npy"), c.kmeans.cluster_centers_)
        else:
            print('Loading saved cluster centers')
            ctrs = np.load(os.path.join(args['LOG_DIR'], f"advanced_centers_{K}.npy"))
            c = getattr(algo, args["CLUSTER"]["TYPE"] + "_clusterer")(TIMES=args["CLUSTER"]["TIMES"], K=K, TOL=1e-4)
            c.fit(tr_stacked[:K])
            c.kmeans.cluster_centers_ = ctrs

        # infer on training set and save
        y = np.concatenate([np.ones((l,)) * i for i, l in enumerate(tr_len_container)], axis=0)
        s = np.concatenate([np.arange(l) for i, l in enumerate(tr_len_container)], axis=0)
        tr_res_df = pd.DataFrame(y, columns=["y"])  # from which sequence
        cluster_l = c.get_assignment(tr_stacked)  # assigned to which cluster
        tr_res_df['cluster'] = cluster_l
        tr_res_df['frame_index'] = s  # the frame index in home sequence
        tr_res_df['seq_name'] = np.concatenate([[name] * tr_len_container[i] for i, name in enumerate(tr_name_container)], axis=0)
        tr_res_df['feat_vec'] = [[vec] for vec in tr_stacked]
        tr_res_df['dist'] = tr_res_df[['feat_vec', 'cluster']].apply(
            lambda x: plain_distance(x['feat_vec'][0],c.kmeans.cluster_centers_[x['cluster']]), #euclidean distance from cluster center
            axis=1)

        proxy_centers_tr = tr_res_df.loc[tr_res_df.groupby('cluster')['dist'].idxmin()].reset_index(drop=True)  #frames with feature vectors closest to cluster centers
        proxy_centers_tr['keypoints3d'] = proxy_centers_tr[['frame_index','seq_name']].apply(
            lambda x: official_loader.load_keypoint3d(x['seq_name'])[x['frame_index']], axis=1)   #3d skeleton keypoints of the closest frame

        sorted_proxies_tr = tr_res_df.groupby('cluster').apply(lambda x: x.sort_values('dist')) #frames in sorted order of closeness to cluster center

        tr_word_df = pd.DataFrame(columns=["idx", "cluster", "length", "y", "name"])  # word index in home sequence
        for sequence_idx in range(len(tr_len_container)):
            name = tr_name_container[sequence_idx]
            cluster_seq = list(cluster_l[tr_where_to_cut[sequence_idx]: tr_where_to_cut[sequence_idx + 1]]) + [-1, ]
            running_idx = 0
            prev = -1
            current_len = 0
            for cc in cluster_seq:
                if cc == prev:
                    current_len += 1
                else:
                    tr_word_df = tr_word_df.append(
                        {"idx": int(running_idx), "cluster": prev, "length": current_len, "y": sequence_idx,
                         "name": name}, ignore_index=True)
                    running_idx += 1
                    current_len = 1
                prev = cc
        tr_word_df = tr_word_df[tr_word_df["idx"] > 0]

        tr_word_df.to_pickle(Path(args['LOG_DIR']) / f"advanced_tr_{K}.pkl")
        print(f"advanced_tr_{K}.pkl dumped to {args['LOG_DIR']}")  # saved tokenization of training set

        tr_res_df.to_pickle(Path(args['LOG_DIR']) / f"advanced_tr_res_{K}.pkl")
        print(f"advanced_tr_res_{K}.pkl dumped to {args['LOG_DIR']}") # frame wise tokenization

        proxy_centers_tr.to_pickle(Path(args['LOG_DIR']) / f"proxy_centers_tr_complete_{K}.pkl")
        print(f"proxy_centers_tr_complete_{K}.pkl dumped to {args['LOG_DIR']}") # saved complete proxy cluster center info
        proxy_centers_tr[['cluster', 'keypoints3d']].to_pickle(Path(args['LOG_DIR']) / f"proxy_centers_tr_{K}.pkl")
        print(f"proxy_centers_tr_{K}.pkl dumped to {args['LOG_DIR']}") # saved proxy centers to feature vector mapping

        sorted_proxies_tr.to_pickle(Path(args['LOG_DIR']) / f"sorted_proxies_tr_{K}.pkl")
        print(f"sorted_proxies_tr_{K}.pkl dumped to {args['LOG_DIR']}") # saved sorted proxies

        # infer on validation set and save
        y = np.concatenate([np.ones((l,)) * i for i, l in enumerate(val_len_container)], axis=0)
        s = np.concatenate([np.arange(l) for i, l in enumerate(val_len_container)], axis=0)
        val_res_df = pd.DataFrame(y, columns=["y"])  # from which sequence
        cluster_l = c.get_assignment(val_stacked)  # assigned to which cluster
        val_res_df['cluster'] = cluster_l
        val_res_df['frame_index'] = s  # the frame index in home sequence
        val_res_df['seq_name'] = np.concatenate([[name] * val_len_container[i] for i, name in enumerate(val_name_container)], axis=0)
        val_res_df['feat_vec'] = [[vec] for vec in val_stacked]
        val_res_df['dist'] = val_res_df[['feat_vec', 'cluster']].apply(
            lambda x: plain_distance(x['feat_vec'][0],c.kmeans.cluster_centers_[x['cluster']]), #euclidean distance from cluster center
            axis=1)
        proxy_centers_val = val_res_df.loc[val_res_df.groupby('cluster')['dist'].idxmin()].reset_index(drop=True)  #frames with feature vectors closest to cluster centers
        proxy_centers_val['keypoints3d'] = proxy_centers_val[['frame_index','seq_name']].apply(
            lambda x: official_loader.load_keypoint3d(x['seq_name'])[x['frame_index']], axis=1)   #3d skeleton keypoints of the closest frame

        sorted_proxies_val = val_res_df.groupby('cluster').apply(lambda x: x.sort_values('dist')) #frames in sorted order of closeness to cluster center

        val_word_df = pd.DataFrame(columns=["idx", "cluster", "length", "y", "name"])  # word index in home sequence
        for sequence_idx in range(len(val_len_container)):
            name = val_name_container[sequence_idx]
            cluster_seq = list(cluster_l[val_where_to_cut[sequence_idx]: val_where_to_cut[sequence_idx + 1]]) + [-1, ]
            running_idx = 0
            prev = -1
            current_len = 0
            for cc in cluster_seq:
                if cc == prev:
                    current_len += 1
                else:
                    val_word_df = val_word_df.append(
                        {"idx": int(running_idx), "cluster": prev, "length": current_len, "y": sequence_idx,
                         "name": name}, ignore_index=True)
                    running_idx += 1
                    current_len = 1
                prev = cc
        val_word_df = val_word_df[val_word_df["idx"] > 0]

        val_word_df.to_pickle(Path(args['LOG_DIR']) / f"advanced_val_{K}.pkl")
        print(f"advanced_val_{K}.pkl dumped to {args['LOG_DIR']}")  # saved tokenization of validation set

        # not needed
        val_res_df.to_pickle(Path(args['LOG_DIR']) / f"advanced_val_res_{K}.pkl")
        print(f"advanced_val_res_{K}.pkl dumped to {args['LOG_DIR']}") # frame wise tokenization

        # not needed
        proxy_centers_val.to_pickle(Path(args['LOG_DIR']) / f"proxy_centers_val_complete_{K}.pkl")
        print(f"proxy_centers_val_complete_{K}.pkl dumped to {args['LOG_DIR']}") # saved proxy centers
        proxy_centers_val[['cluster', 'keypoints3d']].to_pickle(Path(args['LOG_DIR']) / f"proxy_centers_val_{K}.pkl")
        print(f"proxy_centers_val_{K}.pkl dumped to {args['LOG_DIR']}")

        # not needed
        sorted_proxies_val.to_pickle(Path(args['LOG_DIR']) / f"sorted_proxies_val_{K}.pkl")
        print(f"sorted_proxies_val_{K}.pkl dumped to {args['LOG_DIR']}") # saved sorted proxies

if __name__ == '__main__':
    main()
