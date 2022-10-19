#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import os
import subprocess
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import groupby, product
import uuid

import hydra  # https://hydra.cc/docs/intro/
from omegaconf import DictConfig, OmegaConf  # https://github.com/omry/omegaconf
from benedict import benedict as bd  # https://github.com/fabiocaccamo/python-benedict

import pdb

import utils
import viz
from viz import ground_truth_construction as gt_const
from viz import reconstruction

sys.path.append('packages/TEMOS')
import sample_asymov_for_viz


"""
Class that aggregates all visualization functions and data required by them.
"""


class Viz:

    def __init__(self, cfg_rootdir='packages/TEMOS/configs', cfg_name='viz'):
        '''

        Example:
            >>> cfg_p = '~/asymov/...'
            >>> viz_obj = Viz(cfg_p)
        '''
        # Load config from path defined in global var CFG_PATH. # TODO: Arg.
        self.cfg = utils.read_cfg_w_omega(cfg_rootdir, cfg_name)
        print(self.cfg)

        # Load data-structures required for viz.
        self.data = {}
        for f in self.cfg.data_fnames:
            fp = str(Path(self.cfg.datapath, self.cfg.data_fnames[f]))
            self.data[f] = utils.read_pickle(fp)

        # Init. member vars
        self.l_samples = []
        self.n_samples = -1


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
            seq = viz.downsample(self.data['gt'][sid][:5000], ds_ratio)

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
            kp = viz.downsample(kp, 0.3)
            keypoints.append(kp)

        # Visualize
        viz.viz_l_seqs(self.l_samples, keypoints, Path(self.cfg.viz_dir, 'temos_bl'), 'kitml_temos', 25.0)


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

        # Get a list of samples to viz.
        l_samples = kwargs['l_samples'] if 'l_samples' in kwargs.keys() else []
        n_samples = kwargs['n_samples'] if 'n_samples' in kwargs.keys() else -1
        self.l_samples = self._get_l_samples(l_samples, n_samples)
        self.n_samples = len(self.l_samples)

        print(f'Viz. the following {self.n_samples}: {self.l_samples}.')

        # Viz. GT seqs.
        if self.cfg.approaches.gt:
            frames_dir = str(Path(self.cfg.viz_dir, 'gt'))
            gt_const(self.l_samples, self.data['gt'], 'kitml', 0.25, 25.0,
                        frames_dir, force=False)

        # Reconstruct with GT Cluster ids
        if self.cfg.approaches.gt_clid:
            seq2clid_df = self._create_seq2clid_df_gt_cl()  # GT cl ids for seqs.
            self.viz_diff_rec_types(seq2clid_df, 'gt_cluster_recon')

        # Create temp file in kit-splits that sample.py can load.
        # Debug: self.split_file_p = Path('packages/TEMOS/datasets/kit-splits/0d6f926b-52e9-4786-a1ae-1bd7dcf8d592')
        self.split_file_p = Path('packages/TEMOS/datasets/kit-splits', str(uuid.uuid4()))
        utils.write_textf('\n'.join(self.l_samples), self.split_file_p)
        print('Created input seq. list file: ', self.split_file_p)

        # Reconstruct with TEMOS-ASyMov model predictions
        if self.cfg.approaches.asymov_temos:
            self.sample_temos_asymov()  # Get pred cl ids foj

        if self.cfg.approaches.temos_bl:
            self.sample_temos_bl()

        return 1


if __name__ == '__main__':
    '''
    Tests:
    from visualize import Viz
    '''
    viz_obj = Viz()
    # viz_obj.viz_seqs(n_samples=10)
    # viz_obj.viz_seqs(n_samples=2)
    # viz_obj.viz_seqs(l_samples=['00002', '45', 2343, 9999])
    # viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=2)
    # viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=2)

    # TODO: Enable directly giving split name in viz.yaml
    filepath = Path(viz_obj.cfg.path, 'datasets/kit-splits-tiny/visu')
    l_samples = utils.read_textf(filepath, ret_type='list')
    viz_obj.viz_seqs(l_samples=l_samples)

