#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import os, sys
import os.path as osp
from os.path import join as ospj
from os.path import basename as ospb
import pdb
from tqdm import tqdm
import pickle
from multiprocessing import cpu_count, Pool, Process
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import random
import numpy as np
import pandas as pd
import math
import torch
from torch.nn.functional import interpolate as intrp

import subprocess
import shutil
# import wandb
import uuid
import cv2
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter1d, spline_filter1d

from pathlib import Path
import utils
from itertools import groupby, product
import hydra  # https://hydra.cc/docs/intro/
from omegaconf import DictConfig, OmegaConf  # https://github.com/omry/omegaconf
# from benedict import benedict as bd  # https://github.com/fabiocaccamo/python-benedict

#TODO: change the import path to inside acton package
# sys.path.append('/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/')
# from packages.acton.src.data.dataset import loader,utils

# sys.path.append('/content/drive/Shareddrives/vid tokenization/asymov/packages/Complextext2animation/src/')

# sys.path.append('/content/drive/Shareddrives/vid tokenization/asymov/kit-molan/code/')
# import kitml_utils
# sys.path.append('/content/drive/Shareddrives/vid tokenization/asymov/packages/Complextext2animation/src/common/')


# sys.path.append('packages/TEMOS')
# from  packages.TEMOS import sample_asymov_for_viz

"""
Visualize input and output motion sequences and labels
"""

#joint_names--------------------------------------------------------------------

def get_smpl_joint_names():
    # Joint names from SMPL Wiki
    joint_names_map = {
        0: 'Pelvis',

        1: 'L_Hip',
        4: 'L_Knee',
        7: 'L_Ankle',
        10: 'L_Foot',

        2: 'R_Hip',
        5: 'R_Knee',
        8: 'R_Ankle',
        11: 'R_Foot',

        3: 'Spine1',
        6: 'Spine2',
        9: 'Spine3',
        12: 'Neck',
        15: 'Head',

        13: 'L_Collar',
        16: 'L_Shoulder',
        18: 'L_Elbow',
        20: 'L_Wrist',
        22: 'L_Hand',
        14: 'R_Collar',
        17: 'R_Shoulder',
        19: 'R_Elbow',
        21: 'R_Wrist',
        23: 'R_Hand'}

    # Return all joints except indices 22 (L_Hand), 23 (R_Hand)
    return [joint_names_map[idx] for idx in range(len(joint_names_map)-2)]

def get_kitml_joint_names():
    return [
            'root',     # 0
            'BP',       # 1
            'BT',       # 2
            'BLN',      # 3
            'BUN',      # 4
            'LS',       # 5
            'LE',       # 6
            'LW',       # 7
            'RS',       # 8
            'RE',       # 9
            'RW',       # 10
            'LH',       # 11
            'LK',       # 12
            'LA',       # 13
            'LMrot',    # 14
            'LF',       # 15
            'RH',       # 16
            'RK',       # 17
            'RA',       # 18
            'RMrot',    # 19
            'RF']       # 20

def get_nturgbd_joint_names():
    '''From paper:
    1-base of the spine 2-middle of the spine 3-neck 4-head 5-left shoulder 6-left elbow 7-left wrist 8- left hand 9-right shoulder 10-right elbow 11-right wrist 12- right hand 13-left hip 14-left knee 15-left ankle 16-left foot 17- right hip 18-right knee 19-right ankle 20-right foot 21-spine 22- tip of the left hand 23-left thumb 24-tip of the right hand 25- right thumb
    '''
    # Joint names by AC, based on SMPL names
    joint_names_map = {
        0: 'Pelvis',

        12: 'L_Hip',
        13: 'L_Knee',
        14: 'L_Ankle',
        15: 'L_Foot',

        16: 'R_Hip',
        17: 'R_Knee',
        18: 'R_Ankle',
        19: 'R_Foot',

        1: 'Spine1',
        # 'Spine2',
        20: 'Spine3',
        2: 'Neck',
        3: 'Head',

        # 'L_Collar',
        4: 'L_Shoulder',
        5: 'L_Elbow',
        6: 'L_Wrist',
        7: 'L_Hand',
        21: 'L_HandTip',  # Not in SMPL
        22: 'L_Thumb',  # Not in SMPL

        # 'R_Collar',
        8: 'R_Shoulder',
        9: 'R_Elbow',
        10: 'R_Wrist',
        11: 'R_Hand',
        23: 'R_HandTip',  # Not in SMPL
        24: 'R_Thumb',  # Not in SMPL
    }

    return [joint_names_map[idx] for idx in range(len(joint_names_map))]

def get_coco17_joint_names():
    '''
    From loader.py in acton package
    '''
    return [
        'nose', #0
        'left_eye', 'right_eye', #1,2
        'left_ear', 'right_ear', #3,4
        'left_shoulder', 'right_shoulder', #5,6
        'left_elbow', 'right_elbow', #7,8
        'left_wrist', 'right_wrist', #9,10
        'left_hip', 'right_hip', #11,12
        'left_knee', 'right_knee', #13,14
        'left_ankle', 'right_ankle' #15,16
    ]

#-------------------------------------------------------------------------------

#joint_connectivity-------------------------------------------------------------

def get_smpl_skeleton():
    ''' AC -- change the skeleton ordering so that you traverse
    the joints in the following order: Left lower, Left upper,
    Spine, Neck, Head, Right lower, Right upper.
    '''
    return np.array(
        [
            # Left lower
            [ 0, 1 ],
            [ 1, 4 ],
            [ 4, 7 ],
            [ 7, 10],

            # Left upper
            [ 9, 13],
            [13, 16],
            [16, 18],
            [18, 20],
            # [20, 22],

            # Spinal column
            [ 0, 3 ],
            [ 3, 6 ],
            [ 6, 9 ],
            [ 9, 12],
            [12, 15],

            # Right lower
            [ 0, 2 ],
            [ 2, 5 ],
            [ 5, 8 ],
            [ 8, 11],

            # Right upper
            [ 9, 14],
            [14, 17],
            [17, 19],
            [19, 21],
            # [21, 23],
        ])

def get_kitml_skeleton():
    return np.array([
        # Spine
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],

        # Left upper
        [3, 5],
        [5, 6],
        [6, 7],

        # Left lower
        [0, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],

        # Right upper
        [3, 8],
        [8, 9],
        [9, 10],

        # Right lower
        [0, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20]
    ])

def get_nturgbd_skeleton():
    ''' AC -- change the skeleton ordering so that you traverse
    the joints in the following order: Left lower, Left upper,
    Spine, Neck, Head, Right lower, Right upper.
    '''
    return np.array(
        [
            # Left lower
            [0, 12],
            [12, 13],
            [13, 14],
            [14, 15],

            # Left upper
            [4, 20],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 21],
            [7, 22],  # --> L Thumb

            # Spinal column
            [0, 1],
            [1, 20],
            [20, 2],
            [2, 3],

            # Right lower
            [0, 16],
            [16, 17],
            [17, 18],
            [18, 19],

            # Right upper
            [20, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 24],
            # [24, 11] --> R Thumb

            [21, 22],

            [23, 24],

        ]
    )

def get_coco17_skeleton():
    '''
    From visualizer.py in acton package
    '''
    return np.array([
        #Left lower
        [11, 13], [13, 15],

        #Left upper
        [5, 7], [7, 9],

        #Head
        [0, 1], [0, 2],
        [1, 3], [2, 4],

        #Spine
        [5, 11], [6, 12],
        [5, 6], [11, 12], #bridging left and right

        #Right lower
        [12, 14], [14, 16],

        #Right upper
        [6, 8], [8, 10]
    ])

#-------------------------------------------------------------------------------


def get_joint_colors(joint_names):
    '''Return joints based on a color spectrum. Also, joints on
    L and R should have distinctly different colors.
    '''
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(joint_names))]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    return colors


def calc_angle_from_x(sk):
    '''Given skeleton, calc. angle from x-axis'''
    # Hip bone
    id_l_hip = get_smpl_joint_names().index('L_Hip')
    id_r_hip = get_smpl_joint_names().index('R_Hip')
    pl, pr = sk[id_l_hip], sk[id_r_hip]
    bone = np.array(pr-pl)
    unit_v =  bone / np.linalg.norm(bone)
    # Angle with x-axis
    pdb.set_trace()
    x_ax = np.array([1, 0, 0])
    x_angle = math.degrees(np.arccos(np.dot(x_ax, unit_v)))

    '''
    l_hip_z = seq[0, joint_names.index('L_Hip'), 2]
    r_hip_z = seq[0, joint_names.index('R_Hip'), 2]
    az = 0 if (l_hip_z > zroot and zroot > r_hip_z) else 180
    '''
    if bone[1] > 0:
        x_angle = - x_angle

    return x_angle


def calc_angle_from_y(sk):
    '''Given skeleton, calc. angle from x-axis'''
    # Hip bone
    id_l_hip = get_smpl_joint_names().index('L_Hip')
    id_r_hip = get_smpl_joint_names().index('R_Hip')
    pl, pr = sk[id_l_hip], sk[id_r_hip]
    bone = np.array(pl-pr)
    unit_v =  bone / np.linalg.norm(bone)
    print(unit_v)
    # Angle with x-axis
    pdb.set_trace()
    y_ax = np.array([0, 1, 0])
    y_angle = math.degrees(np.arccos(np.dot(y_ax, unit_v)))

    '''
    l_hip_z = seq[0, joint_names.index('L_Hip'), 2]
    r_hip_z = seq[0, joint_names.index('R_Hip'), 2]
    az = 0 if (l_hip_z > zroot and zroot > r_hip_z) else 180
    '''
    # if bone[1] > 0:
    #    y_angle = - y_angle
    seq_y_proj = bone * np.cos(np.deg2rad(y_angle))
    print('Bone projected onto y-axis: ', seq_y_proj)

    return y_angle


def viz_skeleton(seq, folder_p, sk_type='smpl', radius=1, lcolor='#ff0000', rcolor='#0000ff', action='', debug=False):
    ''' Visualize skeletons for given sequence and store as images.

    Args:
        seq (np.array): Array (frames) of joint positions.
        Size depends on sk_type (see below).
            if sk_type is 'smpl' then assume:
                1. first 3 dims = translation.
                2. Size = (# frames, 69)
            elif sk_type is 'nturgbd', then assume:
                1. no translation.
                2. Size = (# frames, 25, 3)
            elif sk_type == 'kitml_temos'
        folder_p (str): Path to root folder containing visualized frames.
            Frames are dumped to the path: folder_p/frames/*.jpg
        radius (float): Space around the subject?

    Returns:
        Stores skeleton sequence as jpg frames.
    '''
    joint_names = None
    az = 90
    if sk_type=='nturgbd':
        joint_names = get_nturgbd_joint_names()
        kin_chain = get_nturgbd_skeleton()
    elif sk_type=='smpl':
        joint_names = get_smpl_joint_names()
        kin_chain = get_smpl_skeleton()  # NOTE that hands are skipped.
        # Reshape flat pose features into (frames, joints, (x,y,z)) (skip trans)
        seq = seq[:, 3:].reshape(-1, len(joint_names), 3).cpu().detach().numpy()
    elif 'kit' in sk_type:
        joint_names = get_kitml_joint_names()
        kin_chain = get_kitml_skeleton()
        # seq[..., 1] = -seq[..., 1]
        seq = seq[..., [2,1,0]]
        seq = seq[..., [0, 2, 1]]
        az = 60
        radius = 1.2
    elif sk_type=='coco17':
        joint_names = get_coco17_joint_names()
        kin_chain = get_coco17_skeleton()
    else:
        assert NotImplementedError

    n_j = len(joint_names)

    # Get color-spectrum for skeleton
    colors = get_joint_colors(joint_names)
    labels = [(joint_names[jidx[0]], joint_names[jidx[1]]) for jidx in kin_chain]

    xroot, yroot, zroot = 0.0, 0.0, 0.0
    if sk_type=='coco17':
        xroot, yroot, zroot = 0.5*(seq[0,11] + seq[0,12])
        seq=seq-np.array([[[xroot, yroot, zroot]]])
        seq=seq/np.max(np.abs(seq))
    elif sk_type == 'kitml_temos':
        # !inital translation. Subtract all frames by frame 0's pelvis.
        seq -= seq[0, 11]
        # !global translation. Subtract all frames by corresponding pelvis.
        seq -= seq[:,11:12,:]
        # x, y, z --> [-1, 1]
        # seq /= np.max(np.abs(seq), axis=(0,1))  # Normalize each dim. separately.
        seq /= np.max(np.abs(seq))  # Normalize all dim. uniformly.
    else:
        xroot, yroot, zroot = seq[0, 0, 0], seq[0, 0, 1], seq[0, 0, 2]

    # seq = seq - seq[0, :, :]

    # Change viewing angle so that first frame is in frontal pose
    # az = calc_angle_from_x(seq[0]-np.array([xroot, yroot, zroot]))
    # az = calc_angle_from_y(seq[0]-np.array([xroot, yroot, zroot]))

    # Create folder for frames
    if not os.path.exists(osp.join(folder_p, 'frames')):
        os.makedirs(osp.join(folder_p, 'frames'))

    # Viz. skeleton for each frame
    # for t in tqdm(range(seq.shape[0]), desc='frames', position=1, leave=False):
    for t in range(seq.shape[0]):
        # Fig. settings
        fig = plt.figure(figsize=(7, 6)) if debug else \
              plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        if sk_type == 'kitml_temos':
            xroot, yroot, zroot = seq[t,11]
        elif 'kit' in sk_type:
            xroot, yroot, zroot = 0.5*(seq[t,11] + seq[t,12])
        else:
            pass  # Do not update root.  # TODO: Verify this.

        # seq[t] = seq[t] - [xroot, yroot, zroot]

        # More figure settings
        ax.set_title(action)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # xroot, yroot, zroot = seq[t, 0, 0], seq[t, 0, 1], seq[t, 0, 2]

        # pdb.set_trace()
        ax.set_xlim3d(-radius + xroot, radius + xroot)
        ax.set_ylim3d(-radius + yroot, radius + yroot)
        ax.set_zlim3d(-radius + zroot, radius + zroot)

        if True==debug:
            ax.axis('on')
            ax.grid(b=True)
        else:
            ax.axis('off')
            ax.grid(b=None)
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_zticklabels([])

        # ax.view_init(75, az)
        # ax.view_init(elev=20, azim=90+az)

        if sk_type == 'coco17':
            ax.view_init(elev=-90, azim=90)
        else:
            ax.view_init(elev=-20, azim=az)

        if True==debug:
            ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
            pass

        for i, (j1, j2) in enumerate(kin_chain):
            # Store bones
            x = np.array([seq[t, j1, 0], seq[t, j2, 0]])#-xroot
            y = np.array([seq[t, j1, 1], seq[t, j2, 1]])#-yroot
            z = np.array([seq[t, j1, 2], seq[t, j2, 2]])#-zroot
            # Plot bones in skeleton
            ax.plot(x, y, z, c=colors[i], marker='o', linewidth=2, label=labels[i])
            pass

        # pdb.set_trace()
        cv2.waitKey(0)

        # fig.savefig(osp.join(folder_p, 'frames', '{1}_{0}.jpg'.format(t, ii)))
        fig.savefig(osp.join(folder_p, 'frames', '{:05d}.jpg'.format(t)))
        plt.close(fig)

        # Debug statement
        # pdb.set_trace()

def viz_skeleton_mp(seq_folder_p, sk_type, radius): # for multi-processing, can pass only 1 arg
    return viz_skeleton(*seq_folder_p, sk_type, radius)

def write_vid_from_imgs(folder_p, fps):
    '''Collate frames into a video sequence.

    Args:
        folder_p (str): Frame images are in the path: folder_p/frames/<int>.jpg
        fps (float): Output frame rate.

    Returns:
        Output video is stored in the path: folder_p/{seq_id}.mp4
    '''
    sid = ospb(folder_p)
    vid_p = osp.join(folder_p, f'{sid}.mp4')
    cmd = ['ffmpeg', '-r', str(int(fps)), '-i',
                    osp.join(folder_p, 'frames', '%05d.jpg'), '-y', vid_p]
    FNULL = open(os.devnull, 'w')
    retcode = subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    if not 0 == retcode:
        print('*******ValueError(Error {0} executing command: {1}*********'.format(retcode, ' '.join(cmd)))
    shutil.rmtree(osp.join(folder_p, 'frames'))

def joint2vid(name_keypoint, sk_type, frames_dir, fps): #combine viz_skeleton
    name, keypoint = name_keypoint
    folder_p = ospj(frames_dir, name)
    viz_skeleton(keypoint, folder_p, sk_type, 1.2)
    write_vid_from_imgs(folder_p, fps)
    return None


def viz_l_seqs(seq_names, keypoints, frames_dir, sk_type, fps, force=False):
    '''1. Dumps sequence of skeleton images for the given sequence of joints.
    2. Collates the sequence of images into an mp4 video.

    Args:
        seq_names (List(str)): name of sequences to visualize
        # TODO: Change this
        keypoints (np.array([T, num_joints, 3])): Joint positions for each frame
        frames_dir (str): Path to dir where visualizations will be dumped.
        sk_type (str): {'smpl', 'nturgbd','kitml','coco17'}
        fps (float): Output frame rate.
        force (Bool): If True visualizes all sequences even if already exists in frames_dir.
                    Defaults to False, only visualizes incomplete or un-visualized sequences.

    Return:
        None. Path of mp4 video: folder_p/{seq_name}.mp4
    '''
    # Delete folder if exists
    if force:
        _ = [shutil.rmtree(ospj(frames_dir, name)) for name in seq_names if osp.exists(ospj(frames_dir, name))]
        viz_names = seq_names
    else:
        viz_names=[]
        for name in seq_names:
            folder_p = ospj(frames_dir, name)
            if osp.exists(folder_p):
                # Rename folder
                if osp.exists(ospj(folder_p, f'{name}.mp4')):
                    # print(f'Video for {folder_p[-5:]} already exists')
                    continue
                else:
                    # print('Deleting existing folder ', folder_p)
                    shutil.rmtree(folder_p)
            viz_names.append(name)
    print(f'Sequences to visualize: {len(viz_names)}')

    # pdb.set_trace()
    # joint2img_procs = []
    # img2vid_procs = []
    # with tqdm(zip(viz_names, keypoints), desc='joint2img', total=len(viz_names), position=0) as pbar:
    #     for name, keypoint in pbar:
    #         pbar.set_description(desc=f'joint2img {name}')
    #         folder_p = ospj(frames_dir, name)
    #         viz_skeleton(keypoint, folder_p, sk_type, 1.2)
    #       # joint2img_proc = Process(target=viz_skeleton, args=(keypoint, folder_p, sk_type, 1.2))
    #       # joint2img_proc.start()
    #       # joint2img_procs.append(joint2img_proc)

    # with tqdm(enumerate(viz_names), desc='img2vid', total=len(viz_names), position=0) as pbar:
    #     for i, name in pbar:
    #         pbar.set_description(desc=f'img2vid {name}')
    #         folder_p = ospj(frames_dir, name)
    #         write_vid_from_imgs(folder_p, fps)
    #         # joint2img_procs[i].join()
    #         # img2vid_proc = Process(target=write_vid_from_imgs, args=(folder_p, fps))
    #         # img2vid_proc.start()
    #         # img2vid_procs.append(img2vid_proc)
    # _ = [p.join() for p in tqdm(img2vid_procs, 'generating vids')]

    # #option 1: new process for each seq_name (with Pool, concurrent operations)
    n = len(viz_names)

    with Pool(cpu_count()*2) as p:
        chunk_size = 2
        iter = zip(viz_names, keypoints)
        with tqdm(desc='joint2vid', total=n) as pbar:
            _ = [pbar.update() for _ in p.imap_unordered(partial(joint2vid, sk_type=sk_type, frames_dir=frames_dir, fps=fps), iter, chunk_size)]

    # option 2: new process for each seq_name for each operation (with Pool, sequential operations)
    # n = len(viz_names)
    # folders_p = [ospj(frames_dir, name) for name in viz_names]
    # with Pool(cpu_count()*2) as p:
    #     chunk_size = 2
    #     iter = zip(keypoints, folders_p)
    #     with tqdm(desc='joint2img', total=n) as pbar:
    #         _ = [pbar.update() for _ in p.imap_unordered(partial(viz_skeleton_mp, sk_type=sk_type, radius=1.2), iter, chunk_size)]

    #     with tqdm(desc='img2vid', total=n) as pbar:
    #         _ = [pbar.update() for _ in p.imap_unordered(partial(write_vid_from_imgs, fps=fps), folders_p, chunk_size)]

    return None


# def viz_rand_seq(X, Y, dtype, epoch, wb, urls=None,
#                  k=3, pred_labels=None):
#     '''
#     Args:
#         X (np.array): Array (frames) of SMPL joint positions.
#         Y (np.array): Multiple labels for each frame in x \in X.
#         dtype (str): {'input', 'pred'}
#         k (int): # samples to viz.
#         urls (tuple): Tuple of URLs of the rendered videos from original mocap.
#         wb (dict): Wandb log dict.
#     Returns:
#         viz_ds (dict): Data structure containing all viz. info so far.
#     '''
#     import packages.Complextext2animation.src.dataUtils as dutils

#     # `idx2al`: idx --> action label string
#     al2idx = dutils.read_json('data/action_label_to_idx.json')
#     idx2al = {al2idx[k]: k for k in al2idx}

#     # Sample k random seqs. to viz.
#     for s_idx in random.sample(list(range(X.shape[0])), k):
#         # Visualize a single seq. in path `folder_p`
#         folder_p = osp.join('viz', str(uuid.uuid4()))
#         viz_seq(seq=X[s_idx], folder_p=folder_p)
#         title='{0} seq. {1}: '.format(dtype, s_idx)
#         pdb.set_trace()
#         acts_str = ', '.join([idx2al[l] for l in torch.unique(Y[s_idx])])
#         wb[title+urls[s_idx]] = wandb.Video(osp.join(folder_p, 'video.mp4'),
#                                            caption='Actions: '+acts_str)

#         if 'pred' == dtype or 'preds'==dtype:
#             raise NotImplementedError☺

#     print('Done viz. {0} seqs.'.format(k))
#     return wb


def fill_xml_jpos(all_j_names, mmm_seq):
    '''Given joint positions with missing values for some axes, return a filled
    numpy array of joint positions.'''
    # T, Root + 20 joints => 21, xyz
    seq = np.zeros((len(mmm_seq), 21, 3))

    # Get joints in the expected order
    plt_jorder = get_kitml_joint_names()
    c = 0

    # Loop over each joint name for plotting
    for j_i, plt_j in enumerate(plt_jorder):
        # Find the relevant joint axis name and store its position
        for ax_i, ax in enumerate(['x', 'y', 'z']):
            if plt_j=='RM' or plt_j=='LM':
                jax_name = plt_j + ax + '_rot'
            else:
                jax_name = plt_j + ax + '_joint'
            if jax_name in all_j_names:
                i = all_j_names.index(jax_name)
                seq[:, j_i, ax_i] = np.array(mmm_seq)[:, i]
                c += 1
    print(f'Found {c} indices.')
    return seq


# def debug_viz_kitml_seq():
#     '''
#     '''
#     import packages.Complextext2animation.src.data as d
#     import packages.Complextext2animation.src.common.mmm as mmm


#     kitml_fol_p = '/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/dataset/kit-mocap/'

#     # List of seq ids to viz.
#     l_sids = list(range(1, 10))
#     for sid in l_sids:
#         fp = Path(ospj(kitml_fol_p, '{0}_mmm.xml'.format(str(sid).zfill(5))))
#         # seq = [ <list> joint names, <list> joint positions] of length (44, 44*T)
#         # all_j_names, mmm_seq = kitml_utils.parse_motions(fp)[0]
#         # all_j_names, mmm_dict = mmm.parse_motions(fp)[0]
#         # seq = fill_xml_jpos(all_j_names, mmm_seq)
#         # jnames, root_pos, root_rot, values, joint_dict = mmm.mmm2csv(fp)

#         # Get joint position data: (T, J=21, 3)
#         K = d.KITMocap(path2data=kitml_fol_p)
#         xyz_data, skel_obj, joints, root_pos, root_rot = K.mmm2quat(fp)

#         # Normalize by root position in case it's not already:
#         seq = xyz_data - np.transpose(root_pos.numpy(), axes=(1,0,2)) # seq = (T, J, 3)

#         # (J, T, 3) --> (T, J, 3)
#         # seq = values.transpose(1, 0, 2)

#         # TODO: Assign root position value. For now, copy over Pelvis values.
#         # jnames = ['root'] + jnames
#         # seq = np.concatenate((seq[:, 0:1, :], seq), axis=1)
#         # Assume root position = (0, 0, 0)
#         # T, J, _ = seq.shape
#         # seq = np.concatenate((np.zeros((T, 1, 3)), seq), axis=1)

#         # Make sure that joints are in the expected order
#         # plt_jorder = get_kitml_joint_names()
#         # new_idxs = []
#         # for j in plt_jorder:
#         #   new_i = jnames.index(j)
#         #   new_idxs.append(new_i)
#         # seq = seq[:, np.array(new_idxs), :]

#         # Viz. "values" (joint positions?)
#         viz_seq(seq, './test_viz/{}_orig_format'.format(sid), 'kitml', debug=True)

#specific_skeleton_viz----------------------------------------------------------

# def viz_kitml_seq():
#     '''
#     '''
#     import packages.Complextext2animation.src.data as d
#     kitml_fol_p = '/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/dataset/kit-mocap/'

#     # List of seq ids to viz.
#     l_sids = list(range(1, 10))
#     for sid in l_sids:
#         fp = Path(ospj(kitml_fol_p, '{0}_mmm.xml'.format(str(sid).zfill(5))))

#         # Get joint position data: (T, J=21, 3)
#         K = d.KITMocap(path2data=kitml_fol_p)
#         xyz_data, skel_obj, joints, root_pos, root_rot = K.mmm2quat(fp)

#         # Normalize by root position in case it's not already.
#         # seq = (T, J, 3)
#         seq = xyz_data - np.transpose(root_pos.numpy(), axes=(1,0,2))

#         # Viz. "values" (joint positions?)
#         viz_seq(seq, './test_viz/{}'.format(sid), 'kitml', debug=True)

# def viz_aistpp_seq():
#     '''
#     '''
#     d = loader.AISTDataset('/content/drive/Shareddrives/vid tokenization/aistpp_subset/aistplusplus/annotations')
#     seq=d.load_keypoint3d('gWA_sFM_cAll_d27_mWA2_ch21')

#     frames_dir='/content/drive/Shareddrives/vid tokenization/frames2'
#     viz_seq(seq, frames_dir, 'coco17', debug=False)

#-------------------------------------------------------------------------------

#cluster2vid--------------------------------------------------------------------

def cluster2vid(clusters_idx, sk_type, proxy_center_info_path, data_path, frames_dir, gt_downsample_ratio=0.25, fps=25.0, duration=0.5, force=False):
    '''
    Args:
        clusters_idx : cluster indices to visualize
        sk_type (str): {'kitml', 'coco17'}
        proxy_center_info_path : path to pickled Dataframe containing sequence name and frame of proxy centers or the DataFrame itself
        data_path : path to the pickled dictionary containing the per frame ground truth 3d keypoints of skeleton joints of the specified video sequence name
                    or the GT dictionary itself
        frames_dir : Path to root folder that will contain frames folder
        gt_downsample_ratio : Float(desired_fps/gt_fps), should be <=1.0. Default = 0.25 (25/100 for kitml).
        fps (float): Desired frame rate. Default = 25.0 (25 for kitml)
        duration : the duration (in secs) for which the cluster visualization should last
        force : If True, visualize all clusters overwriting existing ones. Defaults to False, visualizing only those whose .mp4 videos do not already exist.

    Return:
        None. Path of mp4 video: frames_dir/{cluster_idx}/video.mp4
    '''
    # from packages.acton.src.data.dataset.loader import KITDataset
    # pdb.set_trace()

    #get GT keypoints to visualize
    if type(data_path) == dict:
        ground_truth_data = data_path
    else:
        with open(data_path, 'rb') as handle:
            ground_truth_data = pickle.load(handle)
    # support frames on each side of center frame
    gt_fps = fps/gt_downsample_ratio
    support_frames_count = int((gt_fps*duration-1)/2) #-1 for center frame

    #get proxy center info
    if type(proxy_center_info_path) == pd.DataFrame:
        proxy_center_info = proxy_center_info_path
    else:
        proxy_center_info = pd.read_pickle(proxy_center_info_path)
    center_frames_idx, seq_names = proxy_center_info.loc[clusters_idx, 'frame_index'], proxy_center_info.loc[clusters_idx, 'seq_name']

    seqs = []
    for cluster_idx, center_frame_idx, seq_name in tqdm(zip(clusters_idx, center_frames_idx, seq_names), desc='clusters', total = len(seq_names)):
        seq_complete = ground_truth_data[seq_name]
        seq = seq_complete[max(0, center_frame_idx-support_frames_count):min(seq_complete.shape[0], center_frame_idx+support_frames_count+1)]
        seqs.append(downsample(seq, gt_downsample_ratio))
    #visualize the required fragment of complete sequence
    viz_l_seqs([str(i) for i in clusters_idx], seqs, ospj(frames_dir, str(cluster_idx)), sk_type, fps, force)

#-------------------------------------------------------------------------------

#cluster_seq2vid----------------------------------------------------------------

def cluster_seq2vid(cluster_seq, cluster2keypoint_mapping_path, frames_dir, sk_type):
    '''
    Maps sequence of clusters to proxy center keypoints and visualizes into an mp4 video.

    Args:
        cluster_seq : Array of cluster indices per frame
        cluster2keypoint_mapping_path : Path to pickled dataframe containing the mapping of cluster to proxy center keypoints
        frames_dir : Path to root folder that will contain frames folder
        sk_type : {'smpl', 'nturgbd', 'kitml', 'coco17'}

    Return:
        None. Path of mp4 video: frames_dir/video.mp4
    '''

    cluster2keypoint = pd.read_pickle(cluster2keypoint_mapping_path)
    skeleton_keypoints = np.array([cluster2keypoint.loc[i,'keypoints3d'] for i in cluster_seq])
    viz_l_seqs(skeleton_keypoints, frames_dir, sk_type, debug=False).join()

#-------------------------------------------------------------------------------

#error_metric-------------------------------------------------------------------
def mpjpe3d(seq_names, pred_keypoints, target_keypoints):
    '''
    seq_names List(str): names of sequences to calculate mpjpe for
    pred_keypoints List([T, num_joints, 3]) : 3d keypoints of predicted skeleton joints for each sequence
    target_keypoints List([T, num_joints, 3]) : 3d keypoints of ground-truth skeleton joints for each sequence
    '''
    # pdb.set_trace()
    mpjpe_per_sequence=[]
    with tqdm(zip(seq_names, pred_keypoints, target_keypoints), desc='calculating mpjpe', total=len(seq_names), position=0, leave=False) as pbar:
        for name, pred_keypoint, target_keypoint in pbar:
            pbar.set_description(desc=f'calculating mpjpe for {name}')

            if pred_keypoint.shape[0]!=target_keypoint.shape[0]:
                # if abs(pred_keypoint.shape[0]-target_keypoint.shape[0]) > 100:
                #     print(f'Reconstruction of {name} had {pred_keypoint.shape[0]-target_keypoint.shape[0]} more frames')
                assert pred_keypoint.shape[0]!=0

            mn = min(pred_keypoint.shape[0], target_keypoint.shape[0])
            mpjpe_per_sequence.append(np.mean(np.sqrt(np.sum((target_keypoint[:mn] - pred_keypoint[:mn]) ** 2, axis=2))))

    return mpjpe_per_sequence
#-------------------------------------------------------------------------------

#upsampling---------------------------------------------------------------------
def upsample(motion, ratio):
    assert ratio >= 1.0
    motion = np.array(motion)

    if ratio == 1.0:
        return motion
    step = int(ratio)
    assert step >= 1

    # Alpha blending => interpolation
    alpha = np.linspace(0, 1, step+1)
    last = np.einsum("l,...->l...", 1-alpha, motion[:-1])
    new = np.einsum("l,...->l...", alpha, motion[1:])

    chuncks = (last + new)[:-1]
    output = np.concatenate(chuncks.swapaxes(1, 0))
    # Don't forget the last one
    output = np.concatenate((output, motion[[-1]]))
    return output

def downsample(motion, ratio):
    assert ratio <= 1.0
    motion = np.array(motion)
    num_frames = len(motion)
    if ratio == 1.0:
        return motion

    step = int(1 / ratio)
    assert step >= 1
    subsampled_frames = np.arange(0, num_frames, step, dtype='int32')
    # Don't forget the last one
    output = motion[subsampled_frames]
    return output
#-------------------------------------------------------------------------------

#reconstruction methods-----------------------------------------------------------------

def reconstruction(recons_type, filters, seq_names, data_path, sk_type, recons_upsample_ratio=2.0, gt_downsample_ratio=0.25, fps=25.0, frames_dir=None, viz_names=None, force=False, **kwargs):
    '''
    Args:
        recons_type (str) : reconstruction technique to be used
        filters (List[str]) : smoothing filters to apply on reconstructions. Use string 'none' for no filter.
        seq_names (List[str]): name of video sequences to reconstruct
        data_path : path to the pickled dictionary containing the per frame ground truth 3d keypoints of skeleton joints of the specified video sequence name
                    or the GT dictionary itself
        sk_type : {'smpl', 'nturgbd', 'kitml', 'coco17'}
        recons_upsample_ratio : Float(desired_fps/recons_fps), should be >=1.0. Default = 2.0 (25/12.5 for kitml).
        gt_downsample_ratio : Float(desired_fps/gt_fps), should be <=1.0. Default = 0.25 (25/100 for kitml).
        fps (float): Output frame rate. Default = 25.0 (25 for kitml)
        frames_dir : Path to root folder that will contain frames folder for visualization. If None, won't create visualization.
        viz_names (List[str]): name of video sequences to visualize. Defaults to 'seq_names' argument. Pass [] to not visualize any.
        force : If True, visualize all viz_names overwriting existing ones. Defaults to False, visualizing only those whose .mp4 videos do not already exist.
        **kwargs: Must contain
            if recons_type == 'naive_no_rep' or 'naive':
                contiguous_frame2cluster_mapping_path : Path to pickled dataframe containing the mapping of contiguous frames in a video to a cluster
                cluster2frame_mapping_path : Path to pickled dataframe containing the mapping of cluster to the proxy center frame (and the video sequence containing it)
            if recons_type == 'very_naive':
                cluster2keypoint_mapping_path : Path to pickled dataframe containing the mapping of cluster to proxy center keypoints
                frame2cluster_mapping_path : Path to pickled dataframe containing the mapping of each frame in a video to a cluster.
                frame2cluster_mapping_dir: Path of directory containing .npy files for TEMOS-asymov variant.
    '''

    assert recons_upsample_ratio>=1.0, "recons_upsample_ratio cannot be less than 1"
    assert gt_downsample_ratio<=1.0, "gt_downsample_ratio cannot be greater than 1"
    # if fps is None:
    #     if sk_type == 'kitml':
    #         fps = 25.0
    #     elif sk_type == 'coco17':
    #         fps = 60.0
    #     else:
    #         fps = 30.0

    print('----------------------------------------------------')
    # print(recons_type+'_reconstruction')
    if type(data_path) == dict:
        ground_truth_data = data_path
    else:
        with open(data_path, 'rb') as handle:
            ground_truth_data = pickle.load(handle)
    gt = [ground_truth_data[name][:5000, :, :] for name in seq_names]
    gt = [downsample(keypoint, gt_downsample_ratio) for keypoint in gt]

    if recons_type == 'naive_no_rep' or recons_type  == 'naive':
        contiguous_frame2cluster_mapping_path = kwargs['contiguous_frame2cluster_mapping_path']
        cluster2frame_mapping_path = kwargs['cluster2frame_mapping_path']

        contiguous_frame2cluster = contiguous_frame2cluster_mapping_path
        if type(contiguous_frame2cluster_mapping_path) != pd.DataFrame:
            contiguous_frame2cluster = pd.read_pickle(contiguous_frame2cluster_mapping_path)
        contiguous_cluster_seqs = [contiguous_frame2cluster[contiguous_frame2cluster['name']==name][['cluster', 'length']].reset_index() for name in seq_names]

        output = eval(recons_type+'_reconstruction')(seq_names, contiguous_cluster_seqs,  ground_truth_data, cluster2frame_mapping_path)
        if recons_type == 'naive_no_rep':
            recons, faulty = output
        else:
            recons = output

    elif recons_type == 'very_naive':
        cluster2keypoint_mapping_path = kwargs['cluster2keypoint_mapping_path']
        if 'frame2cluster_mapping_path' in kwargs.keys():
            frame2cluster_mapping_path = kwargs['frame2cluster_mapping_path']
        else:
            frame2cluster_mapping_path = None
        if 'frame2cluster_mapping_dir' in kwargs.keys():
            frame2cluster_mapping_dir = kwargs['frame2cluster_mapping_dir']
        else:
            frame2cluster_mapping_dir = None

        if frame2cluster_mapping_path is not None:
            frame2cluster = pd.read_pickle(frame2cluster_mapping_path)
            cluster_seqs = [frame2cluster[frame2cluster['seq_name']==name]['cluster'] for name in seq_names]
        elif frame2cluster_mapping_dir is not None:
            cluster_seqs = [np.load(os.path.join(frame2cluster_mapping_dir, f"{name}.npy")) for name in seq_names]
        else:
            ValueError('frame2cluster not given')

        recons = eval(recons_type+'_reconstruction')(seq_names, cluster_seqs, cluster2keypoint_mapping_path)

    # recons and gt in desired fps
    recons = [upsample(keypoint, recons_upsample_ratio) for keypoint in recons]

    mpjpe={}
    print('----------------------------------------------------')
    # print("MPJPE")
    for filter in filters:
        if filter == 'none':
            pass
        elif filter == 'spline':
            recons = [spline_filter1d(keypoint, axis=0) for keypoint in recons]
        elif filter == 'uniform':
            recons = [uniform_filter1d(keypoint, size=int(fps/4), axis=0) for keypoint in recons]
        else :
            raise NameError(f'No such filter {filter}')
        print(f"Using {filter} filter")

        mpjpe_per_sequence=mpjpe3d(seq_names, recons, gt)
        mpjpe_mean = np.mean(mpjpe_per_sequence)
        mpjpe['filter'] = mpjpe_mean
        print(f'{recons_type}_{filter} mpjpe: ', mpjpe_mean)

        if frames_dir is not None:
            if viz_names is None:
                viz_names = seq_names

            if filter == 'none':
                frames_dir_temp = frames_dir / f"{recons_type}"
            else:
                frames_dir_temp = frames_dir / f"{recons_type}_{filter}"

            viz_l_seqs(viz_names, recons, frames_dir_temp, sk_type, fps, force)
        print('----------------------------------------------------')
    print('----------------------------------------------------')
    # if per_seq_score:
    #     return np.mean(mpjpe_per_sequence), mpjpe_per_sequence
    # else:
    #     return np.mean(mpjpe_per_sequence)
    return mpjpe


#TODO: naive_no_rep_reconstruction implementation
def naive_no_rep_reconstruction(seq_names, contiguous_cluster_seqs, ground_truth_data, cluster2frame_mapping_path, verbose=True):
    '''
    Args:
        seq_names : name of video sequences to reconstruct
        contiguous_cluster_seqs : mapping of contiguous frames with identically predicted cluster
        ground_truth_data : ground truth 3d keypoints of skeleton joints
        cluster2frame_mapping_path : Path to pickled dataframe containing the mapping of cluster to the proxy center frame (and the video sequence containing it)

    Retruns:
        The reconstructed keypoints
    '''
    assert len(seq_names) == len(contiguous_cluster_seqs)

    reconstructed_keypoints = []
    faulty = []

    cluster2frame = cluster2frame_mapping_path
    if type(cluster2frame_mapping_path) != pd.DataFrame:
        cluster2frame = pd.read_pickle(cluster2frame_mapping_path)

    with tqdm(zip(seq_names, contiguous_cluster_seqs), desc='naive_no_rep reconstruction', total=len(seq_names), disable=(not verbose)) as pbar:
        for name, contiguous_cluster_seq in pbar:
            pbar.set_description(f'naive_no_rep reconstruction - {name}')
            reconstructed_keypoint = []

            # try:
            for i in range(contiguous_cluster_seq.shape[0]):
                #get contiguous cluster info
                contiguous_cluster = contiguous_cluster_seq.iloc[i]
                cluster, length = contiguous_cluster[['cluster', 'length']]
                #get center frame info
                center_frame = cluster2frame.iloc[cluster]
                center_frame_idx, center_frame_keypoint, center_frame_seq_name = center_frame[['frame_index','keypoints3d','seq_name']]
                center_frame_complete_seq = ground_truth_data[center_frame_seq_name]
                assert np.array_equal(center_frame_keypoint,center_frame_complete_seq[center_frame_idx]), "Incorrect center frame sequence"

                center_frame_complete_seq_len = center_frame_complete_seq.shape[0]
                if length >= center_frame_complete_seq_len:
                    # raise Exception(f'seq name : {name}\ncontiguous_cluster_seq : {i}\n')
                    lb = 0
                else:
                    lb = max(0,center_frame_idx - (length-1)//2) #check left boundary
                    lb = min(center_frame_complete_seq_len-length, lb) #check right boundary
                assert lb>=0, f"{name} - Negative left boundary"
                assert lb<center_frame_complete_seq_len, f"{name} - Exceeding right boundary"

                #reconstruct
                reconstructed_keypoint.append(center_frame_complete_seq[lb:lb+min(length, center_frame_complete_seq_len)])
                assert reconstructed_keypoint[-1].shape[0]==length, f"{name} - {reconstructed_keypoint[-1].shape[0]-length} extra frames i reconstruction"

            reconstructed_keypoints.append(np.concatenate(reconstructed_keypoint, axis=0))
            # assert reconstructed_keypoints[-1].shape[0]==ground_truth_keypoints[-1].shape[0]
            # except :
            #     ground_truth_keypoints.pop()
            #     faulty.append(name)
            #     # print(f'{name} cannot be reconstructed naively (no rep)')

    return reconstructed_keypoints, faulty

def naive_reconstruction(seq_names, contiguous_cluster_seqs, ground_truth_data, cluster2frame_mapping_path, verbose=True):
    '''
    Args:
        seq_names : name of video sequences to reconstruct
        contiguous_cluster_seqs : mapping of contiguous frames with identically predicted cluster
        ground_truth_data : ground truth 3d keypoints of skeleton joints
        cluster2frame_mapping_path : Path to pickled dataframe containing the mapping of cluster to the proxy center frame (and the video sequence containing it)

    Retruns:
        The reconstructed keypoints
    '''
    assert len(seq_names) == len(contiguous_cluster_seqs)
    reconstructed_keypoints = []

    cluster2frame = cluster2frame_mapping_path
    if type(cluster2frame_mapping_path) != pd.DataFrame:
        cluster2frame = pd.read_pickle(cluster2frame_mapping_path)

    with tqdm(zip(seq_names, contiguous_cluster_seqs), desc='naive reconstruction', total=len(seq_names), disable=(not verbose)) as pbar:
        for name, contiguous_cluster_seq in pbar:
            pbar.set_description(f'naive reconstruction - {name}')
            reconstructed_keypoint = []

            for i in range(contiguous_cluster_seq.shape[0]):
                #get contiguous cluster info
                contiguous_cluster = contiguous_cluster_seq.iloc[i]
                cluster, length = contiguous_cluster[['cluster', 'length']]
                #get center frame info
                center_frame = cluster2frame.iloc[cluster]
                center_frame_idx, center_frame_keypoint, center_frame_seq_name = center_frame[['frame_index','keypoints3d','seq_name']]
                center_frame_complete_seq = ground_truth_data[center_frame_seq_name]
                assert np.array_equal(center_frame_keypoint,center_frame_complete_seq[center_frame_idx])

                #calculate left right and center frames
                side_length = (length-1)//2
                l_frames = min(side_length, center_frame_idx)
                r_frames = min(side_length, center_frame_complete_seq.shape[0]-1 - center_frame_idx)
                center_reps = length - l_frames - r_frames

                #reconstruct
                reconstructed_keypoint.append(np.concatenate((
                    center_frame_complete_seq[center_frame_idx-l_frames:center_frame_idx], #left supporting frames
                    np.repeat(np.expand_dims(center_frame_keypoint, axis=0), center_reps, axis=0), #center frames
                    center_frame_complete_seq[center_frame_idx+1:center_frame_idx+r_frames+1]), # right supporting frames
                    axis=0
                ))

            reconstructed_keypoints.append(np.concatenate(reconstructed_keypoint, axis=0))

    return reconstructed_keypoints

def very_naive_reconstruction(seq_names, cluster_seqs, cluster2keypoint_mapping_path, verbose=True):
    '''
    Args:
        seq_names : name of video sequences to reconstruct
        cluster_seqs : the mapping of each frame to a cluster.
        cluster2keypoint_mapping_path : Path to pickled dataframe containing the mapping of cluster to proxy center keypoints

    Retruns:
        The reconstructed keypoints
    '''
    # pdb.set_trace()
    assert len(seq_names) == len(cluster_seqs)

    cluster2keypoint = pd.read_pickle(cluster2keypoint_mapping_path)
    reconstructed_keypoints = []
    with tqdm(zip(seq_names, cluster_seqs), desc='very_naive reconstruction', total=len(seq_names), disable=(not verbose)) as pbar:
        for name, cluster_seq in pbar:
            pbar.set_description(f'very_naive reconstruction - {name}')
            reconstructed_keypoints.append(np.array([cluster2keypoint.loc[i,'keypoints3d'] for i in cluster_seq]))

    return reconstructed_keypoints

def ground_truth_construction(seq_names, data_path, sk_type='kitml', gt_downsample_ratio=0.25, fps=25.0, frames_dir=None, force=False):
    '''
    Constructs original video from ground truth sequences, which are used as reference for mpjpe calculation.

    Args:
        seq_names : name of video sequences to construct and visualize ground truth
        data_path : path to the pickled dictionary containing the per frame ground truth 3d keypoints of skeleton joints of the specified video sequence name
                    or the GT dictionary itself
        sk_type : {'smpl', 'nturgbd', 'kitml', 'coco17'}
        gt_downsample_ratio : Float(desired_fps/gt_fps), should be <=1.0. Default = 0.25 (25/100 for kitml).
        fps (float): Output frame rate. Default = 25.0 (25 for kitml).
        frames_dir : Path to root folder that will contain frames folder for visualization.
        force (Bool): If True visualizes all sequences even if already exists in frames_dir.
                    Defaults to False, only visualizes incomplete or un-visualized sequences.
    Returns:
        None.
        The reconstructed videos are saved in {frames_dir}/{seq_name} as {seq_name}.mp4
    '''
    assert frames_dir is not None, "path to store gt visualizations absent"
    assert gt_downsample_ratio<=1.0, "gt_downsample_ratio cannot be greater than 1"

    if type(data_path) == dict:
        ground_truth_data = data_path
    else:
        with open(data_path, 'rb') as handle:
            ground_truth_data = pickle.load(handle)

    print('----------------------------------------------------')
    #TODO: remove 5000 limit
    gt = [downsample(ground_truth_data[name][:5000, :, :], gt_downsample_ratio)
          for name in tqdm(seq_names, 'Ground Truth construction')]
    print('----------------------------------------------------')

    viz_l_seqs(seq_names, gt, frames_dir, sk_type, fps, force)
    print('----------------------------------------------------')

#aggregate reconstruction methods-----------------------------------------------------------------
"""
Class that aggregates all visualization functions and data required by them.
"""

class Viz:

    def __init__(self, cfg: DictConfig):
        '''

        Example:
            >>> cfg_p = '~/asymov/...'
            >>> viz_obj = Viz(cfg_p)
        '''
        # Load config from path defined in global var CFG_PATH. # TODO: Arg.
        self.cfg = cfg
        # print(self.cfg)

        # Load data-structures required for viz.
        self.data = {}
        for f in self.cfg.data_fnames:
            fp = str(Path(self.cfg.datapath, self.cfg.data_fnames[f]))
            self.data[f] = utils.read_pickle(fp)

        # Init. member vars
        self.l_samples = []
        self.n_samples = -1
        self.og_split_file_p = Path(self.cfg.splitpath, self.cfg.split)
        self.og_l_samples = utils.read_textf(self.og_split_file_p, ret_type='list')
        self.og_n_samples = len(self.og_l_samples)


    def _get_l_samples(self, l_samples, n_samples):
        '''
        Args:
            l_samples <list>: of {<int>, <str>} of sample IDs to be viz'd.
        If empty, random `n_samples` are viz'd.
            n_samples <int>: Number of samples to be viz'd. If `l_samples` is
        empty, then random samples are chosen. If not, first `n_samples`
        elements of l_samples is chosen.
        '''
        # l_samples not given. Viz. random seqs.
        if len(l_samples) == 0:
            if n_samples < 0:
                raise TypeError('Either provide list of samples or # samples.')
            else:
                # Random sample `n_samples` seqs. from total samples
                raise NotImplementedError

                n_samples = len(self.val_sids) \
                    if n_samples > len(self.val_sids) else n_samples
                l_samples = random.sample(self.val_sids, n_samples)

        # Viz. from l_samples
        else:
            if n_samples > 0:
                if len(l_samples) != n_samples:
                    print('Warning: Both `n_samples` and `l_samples` are given\
. Visualizing only first `n_samples` from `l_samples`.')
                    l_samples = l_samples[:n_samples]

        # Format all seq IDs
        l_samples = [utils.int2padstr(sid, 5) for sid in l_samples]
        return l_samples


    def _get_cl_assignment(self, sample):
        '''
        '''
        clid_matrix = np.stack(self.data['clid2kp']['keypoints3d'].to_numpy())
        K = clid_matrix.shape[0]
        diff = np.sum((clid_matrix - sample)**2, axis=(1, 2))
        clid = np.argmin(diff)
        return clid


    def _get_clids_for_seq(self, seq):
        '''Given original seq., return equivalent cluster indices.
        Args:
            seq <np.array> (T, 21, 3): xyz keypoints of 21 joints.
        Return:
            cl_ids <list> (T) of <int>. Vector containing cluster ids.
        '''
        l_clids = [self._get_cl_assignment(frame) for frame in seq]
        return l_clids


    def _compress_l_clids(self, sid, idx, l_clids):
        '''Convert list of clids -> (clid, length). Store in the format
        described in _create_seq2clid_df_gt_cl(.).
        '''
        df_dict = {'name': [], 'idx': [], 'cluster': [], 'length': []}
        for k, g in groupby(l_clids):
            df_dict['name'].append(sid)
            df_dict['idx'].append(idx)
            idx += 1
            df_dict['cluster'].append(k)
            df_dict['length'].append(len(list(g)))
        df = pd.DataFrame(data=df_dict)
        return df, idx


    def _create_seq2clid_df_gt_cl(self):
        '''
        DataFrame format:
        name    idx    cluster    length
1       00009    1      953         13
...       ...    ...    ...         ...
178     00034    16     890         8
        '''
        # Append each entry (idx) to this df
        seq2clid_df = pd.DataFrame()
        idx = 1

        # Loop over each seq.
        for sid in self.l_samples:

            # Downsample 100 fps --> 12.5 fps, restrict to 50 sec.
            ds_ratio = self.cfg.fps.pred_fps / float(self.cfg.fps.gt_fps)
            seq = downsample(self.data['gt'][sid][:5000], ds_ratio)

            # Get list of cluster ids for each frame in seq.
            l_clids = self._get_clids_for_seq(seq)

            # Compress list of cl. IDs --> (cl. IDs, counts). See above.
            df, idx = self._compress_l_clids(sid, idx, l_clids)

            # Append to common dataFrome
            seq2clid_df = pd.concat([seq2clid_df, df])

        return seq2clid_df


    def _create_seq2clid_df_preds(self, l_seq_clids):
        '''
        Args:
            l_seq_clids <list>: of <np.array> of cluster ID <int> for each seq.

        Return:
            seq2clid_df <pd.DataFrame>. Format:
                name    idx    cluster    length
        1       00009    1      953         13
        ...       ...    ...    ...         ...
        178     00034    16     890         8
        '''
        # Append each entry (idx) to this df
        seq2clid_df = pd.DataFrame()
        idx = 1

        # Loop over each seq.
        for sid, seq_clids in zip(self.l_samples, l_seq_clids):

            # Compress list of cl. IDs --> (cl. IDs, counts). See above.
            df, idx = self._compress_l_clids(sid, idx, seq_clids)

            # Append to common dataFrome
            seq2clid_df = pd.concat([seq2clid_df, df])

        return seq2clid_df


    def viz_diff_rec_types(self, seq2clid_df, dir_n):
        '''
        Args:
            seq2clid_df <pd.DataFrame>: Described in _create_seq2clid_df(.).
        Frame-rate of cluster ids @ 12.5 fps.
        '''
        frames_dir = Path(self.cfg.viz_dir, dir_n)

        for rec_type in self.cfg.rec_type:
            us_frac = float(self.cfg.fps.out_fps) / self.cfg.fps.pred_fps
            reconstruction(rec_type, self.cfg.filters, self.l_samples,
                self.data['gt'], 'kitml', us_frac, 1.0, self.cfg.fps.out_fps,
                frames_dir, None,
                False, contiguous_frame2cluster_mapping_path=seq2clid_df,
                cluster2frame_mapping_path=self.data['clid2frame'])


    def recons_viz(self):
        '''
        '''
        samples_dir = Path(self.cfg.path, self.cfg.approaches.recons_viz)
        
        #TODO: use pickle file everywhere
        # Get predicted cluster IDs for all seqs. @ 12.5 fps
        l_seq_clids = []
        for sid in self.l_samples:
            kp = np.array(np.load(f'{samples_dir}/{sid}.npy'), dtype=np.int64)
            l_seq_clids.append(kp)

        # Collate preds into specific compressed dataFrame
        seq2clid_df = self._create_seq2clid_df_preds(l_seq_clids)

        self.viz_diff_rec_types(seq2clid_df, 'asymov_mt')


    def sample_mt_asymov(self):
        '''
        Eg., path for model predictions (npy files):
        packages/TEMOS/outputs/kit-xyz-motion-word/asymov_full_run_1/uoon5wnl/samples/neutral_0ade04bd-954f-49bd-b25f-68f3d1ab8f1a
        '''
        ckpt_p = Path(self.cfg.path, self.cfg.approaches.asymov_mt)

        cmd = ['python', 'sample_asymov_mt.py']

        # Overwrite cfg at configs/sample_asymov_mt.yaml
        cmd.append(f'folder={ckpt_p.parent.parent}')
        cmd.append(f'split={self.split_file_p.name}')
        cmd.append(f'ckpt_path={ckpt_p}')
        print(f'Run: ', ' '.join(cmd))

        # Forward pass, store predictions
        subprocess.call(cmd, cwd=str(self.cfg.path))

        # Covert clids --> frames
        clid2kp = np.array(self.data['clid2kp']['keypoints3d'])

        # Destination npy files
        npy_folder = ckpt_p.parent.parent / 'samples' / f'neutral_{self.split_file_p.name}'

        # Get predicted cluster IDs for all seqs. @ 12.5 fps
        l_seq_clids = []
        for sid in self.l_samples:
            kp = np.array(np.load(f'{npy_folder}/{sid}.npy'), dtype=np.int64)
            l_seq_clids.append(kp)

        # Collate preds into specific compressed dataFrame
        seq2clid_df = self._create_seq2clid_df_preds(l_seq_clids)

        self.viz_diff_rec_types(seq2clid_df, 'asymov_mt')


    def sample_temos_asymov(self):
        '''
        Eg., path for model predictions (npy files):
        packages/TEMOS/outputs/kit-xyz-motion-word/asymov_full_run_1/uoon5wnl/samples/neutral_0ade04bd-954f-49bd-b25f-68f3d1ab8f1a
        '''
        ckpt_p = Path(self.cfg.path, self.cfg.approaches.asymov_temos)

        cmd = ['python', 'sample_asymov.py']

        # Overwrite cfg at configs/sample_asymov.yaml
        cmd.append(f'folder={ckpt_p.parent.parent}')
        cmd.append(f'split={self.split_file_p.name}')
        cmd.append(f'ckpt_name={ckpt_p.name}')
        print(f'Run: ', ' '.join(cmd))

        # Forward pass, store predictions
        subprocess.call(cmd, cwd=str(self.cfg.path))

        # Covert clids --> frames
        clid2kp = np.array(self.data['clid2kp']['keypoints3d'])

        # Destination npy files
        npy_folder = ckpt_p.parent.parent / 'samples' / f'neutral_{self.split_file_p.name}'

        # Get predicted cluster IDs for all seqs. @ 12.5 fps
        l_seq_clids = []
        for sid in self.l_samples:
            kp = np.array(np.load(f'{npy_folder}/{sid}.npy'), dtype=np.int64)
            l_seq_clids.append(kp)

        # Collate preds into specific compressed dataFrame
        seq2clid_df = self._create_seq2clid_df_preds(l_seq_clids)

        self.viz_diff_rec_types(seq2clid_df, 'asymov_temos')


    def sample_temos_bl(self):
        '''Sample and visualize seqs. specified in `self.l_samples`. Use TEMOS's sampling
        and viz. code out of the box.

        For testing purposes:
            <file_p>: packages/TEMOS/datasets/kit-splits/0d6f926b-52e9-4786-a1ae-1bd7dcf8d592
        '''

        # Load "latest ckpt" from folder spec. in viz.yaml. Overwrite TEMOS's sample cfg.
        folder = Path(self.cfg.path, self.cfg.approaches.temos_bl)

        # TEMOS's sampling script saves pred xyz motions as npy's in ${folder}/samples/${split}
        sample_args = f'folder={folder} split={self.split_file_p.name}'
        os.chdir('packages/TEMOS')
        cmd = f'HYDRA_FULL_ERROR=1 python sample.py {sample_args}'
        print(f'Run: ', cmd)
        os.system(cmd)

        # Destination npy files: 
        npy_folder = folder / 'samples' / f'neutral_{self.split_file_p.name}'
        keypoints = []
        for sid in self.l_samples:
            kp = np.load(f'{npy_folder}/{sid}.npy')
            # Downsample kp. TODO: Verify assumption of 100fps output
            kp = downsample(kp, 0.3)
            keypoints.append(kp)

        # Visualize
        viz_l_seqs(self.l_samples, keypoints, Path(self.cfg.viz_dir, 'temos_bl'), 'kitml_temos', 25.0)


    def viz_seqs(self, **kwargs):
        '''Viz. a list of seqs.

        Example:
            >>> cfg_p = '~/asymov/...'
            >>> viz_obj = Viz(cfg_p)
            >>> viz_obj.viz_seqs(n_samples=10)
            >>> viz_obj.viz_seqs(l_samples=['00002', '45', 2343, 9999])
            >>> viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=1)
            >>> viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=1)
        '''

        #TODO: redundant
        # Get a list of samples to viz.
        l_samples = kwargs['l_samples'] if 'l_samples' in kwargs.keys() else self.og_l_samples
        n_samples = kwargs['n_samples'] if 'n_samples' in kwargs.keys() else self.og_n_samples
        self.l_samples = self._get_l_samples(l_samples, n_samples)
        self.n_samples = len(self.l_samples)

        print(f'Viz. the following {self.n_samples}: {self.l_samples}.')

        # Viz. GT seqs.
        if self.cfg.approaches.gt:
            frames_dir = str(Path(self.cfg.viz_dir, 'gt'))
            ground_truth_construction(self.l_samples, self.data['gt'], 'kitml', 0.25, 25.0,
                        frames_dir, force=False)

        # Reconstruct with GT Cluster ids
        if self.cfg.approaches.gt_clid:
            seq2clid_df = self._create_seq2clid_df_gt_cl()  # GT cl ids for seqs.
            self.viz_diff_rec_types(seq2clid_df, 'gt_cluster_recon')

        # Create temp file in kit-splits that sample.py can load.
        if self.l_samples == self.og_l_samples:
            self.split_file_p = self.og_split_file_p
            print(f'Using given split: {self.cfg.split}')
        else:
            self.split_file_p = Path(self.cfg.splitpath, str(uuid.uuid4()))
            utils.write_textf('\n'.join(self.l_samples), self.split_file_p)
            print('Created input seq. list file: ', self.split_file_p)

        # Reconstruct and visualize
        if self.cfg.approaches.recons_viz:
            self.recons_viz()
        
        # Inference MT-ASyMov model and reconstruct
        if self.cfg.approaches.asymov_mt:
            self.sample_mt_asymov()

        # Inference TEMOS-ASyMov model and reconstruct
        if self.cfg.approaches.asymov_temos:
            self.sample_temos_asymov()  # Get pred cl ids foj

        if self.cfg.approaches.temos_bl:
            self.sample_temos_bl()

        return 1
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    # from packages.acton.src.data.dataset.loader import KITDataset

    # viz_kitml_seq()
    # viz_aistpp_seq()

    # d = pd.read_pickle('/content/drive/Shareddrives/vid tokenization/acton/logs/TAN/advanced_tr_res_150.pkl')
    # cluster_seq = d[d['seq_name']=='gWA_sFM_cAll_d25_mWA2_ch03']['cluster']
    # cluster_seq2vid(cluster_seq[:500], '/content/drive/Shareddrives/vid tokenization/acton/logs/TAN/proxy_centers_tr_150.pkl', '/content/drive/Shareddrives/vid tokenization/seq2vid', 'coco17')

    # cluster2vid(clusters_idx=[i for i in range(150)], sk_type='kitml',
    #     proxy_center_info_path='/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/proxy_centers_tr_complete_150.pkl',
    #     data_dir='/content/drive/Shareddrives/vid tokenization/asymov/kit-molan/', data_name='xyz',
    #     frames_dir='/content/drive/Shareddrives/vid tokenization/cluster2vid')

    data_dir = '/content/drive/Shareddrives/vid tokenization/asymov/kit-molan/'
    data_name = 'xyz'
    from packages.acton.src.data.dataset.loader import KITDataset
    d = KITDataset(data_dir, data_name)

    # seq_names=['02654']
    # seq = d.load_keypoint3d('02654')
    # seq_names = ['01699', #'02855',
    #    '00826', '02031', '01920', '02664', '01834',
    #    '02859', '00398', '03886', '01302', '02053', '00898', '03592',
    #    '03580', '00771', '01498', '00462', '01292', '02441', '03393',
    #    '00376', '02149', '03200', '03052', '01788', '00514', '01744',
    #    '02977', '00243', '02874', '00396', '03597', '02654', '03703',
    #    '00456', '00812', '00979', '00724', '01443', '03401', '00548',
    #    '00905', '00835', #'02612',
    #    '02388', '03788', '03870', '03181',
    #    '00199']
    seq_names = ["00017",
        "00018",
        "00002",
        "00014",
        "00005",
        "00010"]

    #TODO use cofig to get viz paths
    frame2cluster_mapping_path = '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/20220409_173106/advanced_tr_res_150.pkl'
    contiguous_frame2cluster_mapping_path = '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/20220409_173106/advanced_tr_150.pkl'
    cluster2keypoint_mapping_path = '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/20220409_173106/proxy_centers_tr_150.pkl'
    cluster2frame_mapping_path = '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/20220409_173106/proxy_centers_tr_complete_150.pkl'
    sk_type = 'kitml'
    frames_dir = '/content/drive/Shareddrives/vid tokenization/kit_reconstruction/'

    # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, frames_dir+'very_naive')
    very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type)
    print('very naive mpjpe : ', very_naive_mpjpe_mean)

    # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, frames_dir+'naive')
    naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type)
    print('naive mpjpe : ', naive_mpjpe_mean)

    # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, frames_dir+'naive_no_rep')
    naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type)
    print('naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)


    #uniform filter
    # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, filter = 'uniform', frames_dir=frames_dir+'very_naive_ufilter')
    very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, filter='uniform')
    print('uniform filtered very naive mpjpe : ', very_naive_mpjpe_mean)

    # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='uniform', frames_dir=frames_dir+'naive_ufilter')
    naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='uniform')
    print('uniform filtered naive mpjpe : ', naive_mpjpe_mean)

    # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='uniform', frames_dir=frames_dir+'naive_no_rep_ufilter')
    naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='uniform')
    print('uniform filtered naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)

    #spline filter
    # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, filter = 'spline', frames_dir=frames_dir+'very_naive_sfilter')
    very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, filter='spline')
    print('spline filtered very naive mpjpe : ', very_naive_mpjpe_mean)

    # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='spline', frames_dir=frames_dir+'naive_sfilter')
    naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='spline')
    print('spline filtered naive mpjpe : ', naive_mpjpe_mean)

    # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='spline', frames_dir=frames_dir+'naive_no_rep_sfilter')
    naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='spline')
    print('spline filtered naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)
