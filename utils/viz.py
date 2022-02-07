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
import pdb

import random
import numpy as np
import pandas as pd
import math
import torch
from torch.nn.functional import interpolate as intrp

import subprocess
import shutil
import wandb
import uuid
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path

#TODO: change the import path to inside acton package
from src.data.dataset import loader,utils
from tqdm import tqdm
import pdb

# sys.path.append('/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/src/')
# import dataUtils as dutils
# import data as d
# import kitml_utils
# sys.path.append('/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/src/common')
# import mmm


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
    elif sk_type=='kit' or sk_type=='kitml':
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

    # xroot, yroot, zroot = 0.0, 0.0, 0.0
    if sk_type=='coco17':
        xroot, yroot, zroot = 0.5*(seq[0,11] + seq[0,12])
        seq=seq-np.array([[[xroot, yroot, zroot]]])
        seq=seq/np.max(np.abs(seq))
    else:
        xroot, yroot, zroot = seq[0, 0, 0], seq[0, 0, 1], seq[0, 0, 2]
    # seq = seq - seq[0, :, :]

    # Change viewing angle so that first frame is in frontal pose
    # az = calc_angle_from_x(seq[0]-np.array([xroot, yroot, zroot]))
    # az = calc_angle_from_y(seq[0]-np.array([xroot, yroot, zroot]))

    # Viz. skeleton for each frame
    for t in tqdm(range(seq.shape[0]), desc='frames'):
        # pdb.set_trace()
        # Fig. settings
        fig = plt.figure(figsize=(7, 6)) if debug else \
              plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        xroot, yroot, zroot = 0.5*(seq[t,11] + seq[t,12])
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
        ax.view_init(elev=-90, azim=90)
        # ax.view_init(elev=110, azim=70)
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
        fig.savefig(osp.join(folder_p, 'frames', '{0}.jpg'.format(t)))
        plt.close(fig)

        # Debug statement
        # pdb.set_trace()


def write_vid_from_imgs(folder_p, fps):
    '''Collate frames into a video sequence.

    Args:
        folder_p (str): Frame images are in the path: folder_p/frames/<int>.jpg
        fps (float): Output frame rate.

    Returns:
        Output video is stored in the path: folder_p/video.mp4
    '''
    vid_p = osp.join(folder_p, 'video.mp4')
    cmd = ['ffmpeg', '-r', str(int(fps)), '-i',
                    osp.join(folder_p, 'frames', '%d.jpg'), '-y', vid_p]
    FNULL = open(os.devnull, 'w')
    retcode = subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    if not 0 == retcode:
        print('*******ValueError(Error {0} executing command: {1}*********'.format(retcode, ' '.join(cmd)))
    shutil.rmtree(osp.join(folder_p, 'frames'))


def viz_seq(seq, folder_p, sk_type, debug=False):
    '''1. Dumps sequence of skeleton images for the given sequence of joints.
    2. Collates the sequence of images into an mp4 video.

    Args:
        seq (np.array): Array of joint positions.
        folder_p (str): Path to root folder that will contain frames folder.
        sk_type (str): {'smpl', 'nturgbd'}

    Return:
        None. Path of mp4 video: folder_p/video.mp4
    '''
    # pdb.set_trace()

    # Delete folder if exists
    if osp.exists(folder_p):
        print('Deleting existing folder ', folder_p)
        shutil.rmtree(folder_p)

    # Create folder for frames
    os.makedirs(osp.join(folder_p, 'frames'))

    # Dump frames into folder. Args: (data, radius, frames path)
    viz_skeleton(seq, folder_p=folder_p, sk_type=sk_type, radius=1.2, debug=debug)
    if sk_type == 'kitml' or sk_type == 'coco17':
        write_vid_from_imgs(folder_p, 60.0)
    else:
        write_vid_from_imgs(folder_p, 30.0)

    return None


def viz_rand_seq(X, Y, dtype, epoch, wb, urls=None,
                 k=3, pred_labels=None):
    '''
    Args:
        X (np.array): Array (frames) of SMPL joint positions.
        Y (np.array): Multiple labels for each frame in x \in X.
        dtype (str): {'input', 'pred'}
        k (int): # samples to viz.
        urls (tuple): Tuple of URLs of the rendered videos from original mocap.
        wb (dict): Wandb log dict.
    Returns:
        viz_ds (dict): Data structure containing all viz. info so far.
    '''
    # `idx2al`: idx --> action label string
    al2idx = dutils.read_json('data/action_label_to_idx.json')
    idx2al = {al2idx[k]: k for k in al2idx}

    # Sample k random seqs. to viz.
    for s_idx in random.sample(list(range(X.shape[0])), k):
        # Visualize a single seq. in path `folder_p`
        folder_p = osp.join('viz', str(uuid.uuid4()))
        viz_seq(seq=X[s_idx], folder_p=folder_p)
        title='{0} seq. {1}: '.format(dtype, s_idx)
        pdb.set_trace()
        acts_str = ', '.join([idx2al[l] for l in torch.unique(Y[s_idx])])
        wb[title+urls[s_idx]] = wandb.Video(osp.join(folder_p, 'video.mp4'),
                                           caption='Actions: '+acts_str)

        if 'pred' == dtype or 'preds'==dtype:
            raise NotImplementedError

    print('Done viz. {0} seqs.'.format(k))
    return wb


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


def debug_viz_kitml_seq():
    '''
    '''
    kitml_fol_p = '/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/dataset/kit-mocap/'

    # List of seq ids to viz.
    l_sids = list(range(1, 10))
    for sid in l_sids:
        fp = Path(ospj(kitml_fol_p, '{0}_mmm.xml'.format(str(sid).zfill(5))))
        # seq = [ <list> joint names, <list> joint positions] of length (44, 44*T)
        # all_j_names, mmm_seq = kitml_utils.parse_motions(fp)[0]
        # all_j_names, mmm_dict = mmm.parse_motions(fp)[0]
        # seq = fill_xml_jpos(all_j_names, mmm_seq)
        # jnames, root_pos, root_rot, values, joint_dict = mmm.mmm2csv(fp)

        # Get joint position data: (T, J=21, 3)
        K = d.KITMocap(path2data=kitml_fol_p)
        xyz_data, skel_obj, joints, root_pos, root_rot = K.mmm2quat(fp)

        # Normalize by root position in case it's not already:
        seq = xyz_data - np.transpose(root_pos.numpy(), axes=(1,0,2)) # seq = (T, J, 3)

        # (J, T, 3) --> (T, J, 3)
        # seq = values.transpose(1, 0, 2)

        # TODO: Assign root position value. For now, copy over Pelvis values.
        # jnames = ['root'] + jnames
        # seq = np.concatenate((seq[:, 0:1, :], seq), axis=1)
        # Assume root position = (0, 0, 0)
        # T, J, _ = seq.shape
        # seq = np.concatenate((np.zeros((T, 1, 3)), seq), axis=1)

        # Make sure that joints are in the expected order
        # plt_jorder = get_kitml_joint_names()
        # new_idxs = []
        # for j in plt_jorder:
        #   new_i = jnames.index(j)
        #   new_idxs.append(new_i)
        # seq = seq[:, np.array(new_idxs), :]

        # Viz. "values" (joint positions?)
        viz_seq(seq, './test_viz/{}_orig_format'.format(sid), 'kitml', debug=True)

#specific_skeleton_viz----------------------------------------------------------

def viz_kitml_seq():
    '''
    '''
    kitml_fol_p = '/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/dataset/kit-mocap/'

    # List of seq ids to viz.
    l_sids = list(range(1, 10))
    for sid in l_sids:
        fp = Path(ospj(kitml_fol_p, '{0}_mmm.xml'.format(str(sid).zfill(5))))

        # Get joint position data: (T, J=21, 3)
        K = d.KITMocap(path2data=kitml_fol_p)
        xyz_data, skel_obj, joints, root_pos, root_rot = K.mmm2quat(fp)

        # Normalize by root position in case it's not already.
        # seq = (T, J, 3)
        seq = xyz_data - np.transpose(root_pos.numpy(), axes=(1,0,2))

        # Viz. "values" (joint positions?)
        viz_seq(seq, './test_viz/{}'.format(sid), 'kitml', debug=True)

def viz_aistpp_seq():
    '''
    '''
    d = loader.AISTDataset('/content/drive/Shareddrives/vid tokenization/aistpp_subset/aistplusplus/annotations')
    seq=d.load_keypoint3d('gWA_sFM_cAll_d27_mWA2_ch21')

    frames_dir='/content/drive/Shareddrives/vid tokenization/frames2'
    viz_seq(seq, frames_dir, 'coco17', debug=False)

#-------------------------------------------------------------------------------

#cluster2vid--------------------------------------------------------------------

def aistpp_cluster2vid(cluster_idx, proxy_center_info_path, frames_dir, support_frames_count=20):
    '''
    cluster_idx : cluster index to visualize
    proxy_center_info_path : path to pickled Dataframe containing sequence name and frame of proxy centers
    support_frames_count : maximum number of frames before and after the one corresponding proxy center, defaults to 60
    frames_dir : Path to root folder that will contain frames folder

    Return:
        None. Path of mp4 video: frames_dir/video.mp4
    '''

    #get proxy center info
    proxy_center_info = pd.read_pickle(proxy_center_info_path)
    center_frame_idx, seq_name = proxy_center_info.loc[cluster_idx, ['frame_index', 'seq_name']]

    #get keypoints to visualize
    #TODO : change the path to official loader
    d = loader.AISTDataset('/content/drive/Shareddrives/vid tokenization/aistpp_subset/aistplusplus/annotations')
    seq_complete = d.load_keypoint3d(seq_name)
    seq = seq_complete[max(0, center_frame_idx-support_frames_count):min(seq_complete.shape[0], center_frame_idx+support_frames_count+1)]
    
    #visualize the required fragment of complete sequence
    viz_seq(seq, frames_dir, 'coco17', debug=False)

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
    viz_seq(skeleton_keypoints, frames_dir, sk_type, debug=False)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    # viz_kitml_seq()
    # viz_aistpp_seq()
    
    # d = pd.read_pickle('/content/drive/Shareddrives/vid tokenization/acton/logs/TAN/advanced_tr_res_150.pkl')
    # cluster_seq = d[d['seq_name']=='gWA_sFM_cAll_d25_mWA2_ch03']['cluster']
    # cluster_seq2vid(cluster_seq[:500], '/content/drive/Shareddrives/vid tokenization/acton/logs/TAN/proxy_centers_tr_150.pkl', '/content/drive/Shareddrives/vid tokenization/seq2vid', 'coco17')

    aistpp_cluster2vid(0, '/content/drive/Shareddrives/vid tokenization/acton/logs/TAN/proxy_centers_tr_complete_150.pkl', '/content/drive/Shareddrives/vid tokenization/cluster2vid')

