#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import os
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import groupby, product

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

		# TODO: self.n_val_sids = list(val_samples.keys())
		self.val_sids = ['00002', '00023', '00089', '00003']

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

				# TODO: Remove next line once self.val_sids is implemented
				n_samples = len(self.val_sids) \
					if n_samples > len(self.val_sids) else n_samples
				l_samples = random.sample(self.val_sids, n_samples)

				# TODO: Support random train viz.

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


	def _create_seq2clid_df(self):
		'''
		DataFrame format:
	    name    idx    cluster    length
1    	00009    1    	953    		13
...  	  ...    ...    ...    		...
178    	00034    16    	890    		8
		'''
		df_ds = {'name': [], 'idx': [], 'cluster': [], 'length': []}
		idx = 1

		# Loop over each seq.
		for sid in self.l_samples:

			# Downsample 100 fps --> 12.5 fps, restrict to 50 sec.
			ds_ratio = self.cfg.fps.pred_fps / float(self.cfg.fps.gt_fps)
			seq = viz.downsample(self.data['gt'][sid][:5000], ds_ratio)

			# Get list of cluster ids for each frame in seq.
			l_clids = self._get_clids_for_seq(seq)

			# Convert list of clids -> (clid, length)
			for k, g in groupby(l_clids):
				df_ds['name'].append(sid)
				df_ds['idx'].append(idx)
				idx += 1
				df_ds['cluster'].append(k)
				df_ds['length'].append(len(list(g)))

		seq2clid_df = pd.DataFrame(data=df_ds)
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
				self.data['gt'], 'kitml', us_frac, 0.0, self.cfg.fps.out_fps,
				frames_dir, None,
				False, contiguous_frame2cluster_mapping_path=seq2clid_df,
				cluster2frame_mapping_path=self.data['clid2frame'])


	def sample_temos_asymov(self):
		'''
folder = "'outputs/kit-xyz-motion-word/asymov_legit/382wpogn'"
ckpt_name="'latest-epoch\=177.ckpt'"
split="'visu'"
!HYDRA_FULL_ERROR=1 python sample_asymov.py folder=$folder ckpt_name=$ckpt_name split=$split
		'''
		ckpt_p = Path(self.cfg.path, self.cfg.approaches.asymov_temos)

		# execute
		cs_str = ','.join(self.l_samples)
		args = f'ckpt_p={ckpt_p} l_samples=[{cs_str}] viz_dir={self.cfg.viz_dir}'
		cmd = f'HYDRA_FULL_ERROR=1 python packages/TEMOS/sample_asymov_for_viz.py {args}'
		print('Running: ', cmd)
		os.system(cmd)
		pdb.set_trace()

		# Relies on cfg at configs/sample_asymov.yaml


	def viz_seqs(self, **kwargs):
		'''Viz. a list of seqs.

		Example:
			>>> cfg_p = '~/asymov/...'
			>>> viz_obj = Viz(cfg_p)
			>>> viz_obj.viz_seqs(n_samples=10)
			>>> viz_obj.viz_seqs(l_samples=['00002', '45', 2343, 9999])
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
			seq2clid_df = self._create_seq2clid_df()  # GT cl ids for seqs.
			self.viz_diff_rec_types(seq2clid_df, 'gt_cluster_recon')

		# Reconstruct with TEMOS-ASyMov model predictions
		if self.cfg.approaches.asymov_temos:
			seq2clid_df = self.sample_temos_asymov()  # Get pred cl ids foj
			pdb.set_trace()

			# TODO:
			# self.viz_diff_rec_types(seq2clid_df, 'temos_asymov_recon')

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
	viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=2)


