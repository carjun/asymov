from typing import List
from pathlib import Path
import pickle
import os
import pandas as pd
import numpy as np
import pdb

import torch
from torch import Tensor, nn
from torchmetrics import Metric, MeanMetric
from hydra.utils import instantiate

from temos.transforms.joints2jfeats import Rifke
from temos.tools.geometry import matrix_of_angles
from temos.model.utils.tools import remove_padding
import sys
# pdb.set_trace()
sys.path.append('../../../../../../')
from viz import very_naive_reconstruction, naive_reconstruction, naive_no_rep_reconstruction, mpjpe3d, upsample, downsample
from scipy.ndimage import uniform_filter1d, spline_filter1d

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)

class Perplexity(MeanMetric):
    '''
    Calculates perplexity from logits and target.
    Wrapper around SumMetric.
    '''
    def __init__(self, ignore_index: int, **kwargs):
        super().__init__(**kwargs)
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def update(self, logits: Tensor, target: Tensor):
        # Compute $$\sum PP}L(X)$$ where X = single sequence.
        # Since target = 1-hot, we use CE(.) to compute -log p_{gt}
        # pdb.set_trace()
        ce_tensor = self.CE(logits, target) #[b_sz, T]
        ppl = torch.exp(ce_tensor.mean(dim=-1)) #[b_sz]
        return super().update(ppl)

class ReconsMetrics(Metric):
    def __init__(self, recons_types: List[str], filters: List[str], gt_path: str,
                 fps: float, recons_fps: float, gt_fps: float, num_mw_clusters: int,
                #  jointstype: str = "mmm",
                #  force_in_meter: bool = True,
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.recons_types = recons_types
        self.filters = filters
        self.recons_upsample_ratio=fps/recons_fps
        self.gt_downsample_ratio=fps/gt_fps
        self.fps = fps
        self.num_clusters = num_mw_clusters
        self.kwargs = kwargs
        
        gt_path = Path(gt_path)
        print("Retrieving GT data for recons loss from", gt_path)
        with open(gt_path, 'rb') as handle:
            self.ground_truth_data = pickle.load(handle)
        
        
        # if jointstype != "mmm":
        #     raise NotImplementedError("This jointstype is not implemented.")

        # super().__init__()
        # self.jointstype = jointstype
        # self.rifke = Rifke(jointstype=jointstype,
        #                    normalization=False)
        
        # self.force_in_meter = force_in_meter
        # self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_good_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # APE
        # self.add_state("APE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.add_state("APE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.add_state("APE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        # self.add_state("APE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        # self.APE_metrics = ["APE_root", "APE_traj", "APE_pose", "APE_joints"]

        # AVE
        # self.add_state("AVE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.add_state("AVE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.add_state("AVE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        # self.add_state("AVE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        # self.AVE_metrics = ["AVE_root", "AVE_traj", "AVE_pose", "AVE_joints"]

        # MPJPE
        self.metrics=[]
        for recons_type in self.recons_types:
            for filter in self.filters:
                self.add_state(f"MPJPE_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.metrics.append(f"MPJPE_{recons_type}_{filter}")
        # All metric
        # self.metrics = self.APE_metrics + self.AVE_metrics
        
    def compute(self):
        # count = self.count
        # APE_metrics = {metric: getattr(self, metric) / count for metric in self.APE_metrics}

        # # Compute average of APEs
        # APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        # APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count

        count_seq = self.count_good_seq
        recons_metrics = {metric: getattr(self, metric) / count_seq for metric in self.metrics}
        # AVE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.AVE_metrics}

        # # Compute average of AVEs
        # AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        # AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq

        return {**recons_metrics, 'all_seq':self.count_seq, 'good_seq':self.count_good_seq, 
                'good_seq_%': self.count_good_seq/self.count_seq}
        # return {**APE_metrics, **AVE_metrics}

    def update(self, seq_names: List[str], cluster_seqs: List[Tensor]):
        assert len(seq_names)==len(cluster_seqs)
        # self.count += sum(lengths)
        self.count_seq += len(seq_names)
        good_idx = [i for i,cluster_seq in enumerate(cluster_seqs) \
            if cluster_seq.max()<self.num_clusters and cluster_seq.min()>=0]
        
        seq_names = [seq_names[i] for i in good_idx]
        cluster_seqs = [cluster_seqs[i] for i in good_idx]
        self.count_good_seq += len(seq_names)
        
        gt = [self.ground_truth_data[name][:5000, :, :] for name in seq_names]
        gt = [downsample(keypoint, self.gt_downsample_ratio) for keypoint in gt]
        cluster_seqs = [cluster_seq.numpy() for cluster_seq in cluster_seqs]
        
        if ('naive_no_rep' in self.recons_types) or ('naive' in self.recons_types):
            contiguous_frame2cluster_mapping = {"name":[], "idx":[], "cluster":[], "length":[]}
            for name, cluster_seq in zip(seq_names, cluster_seqs):
                prev=-1
                running_idx=0
                current_len = 0
                cluster_seq = np.append(cluster_seq, [-1])
                for cc in cluster_seq:
                    if cc == prev:
                        current_len += 1
                    else:
                        contiguous_frame2cluster_mapping["name"].append(name)
                        contiguous_frame2cluster_mapping["idx"].append(int(running_idx))
                        contiguous_frame2cluster_mapping["cluster"].append(prev)
                        contiguous_frame2cluster_mapping["length"].append(current_len)
                        running_idx += 1
                        current_len = 1
                    prev = cc
            contiguous_frame2cluster_mapping = pd.DataFrame.from_dict(contiguous_frame2cluster_mapping)
            contiguous_frame2cluster_mapping = contiguous_frame2cluster_mapping[contiguous_frame2cluster_mapping["idx"]>0]
            contiguous_cluster_seqs = [contiguous_frame2cluster_mapping[contiguous_frame2cluster_mapping['name']==name][['cluster', 'length']].reset_index(drop=True) for name in seq_names]

        # pdb.set_trace()
        for recons_type in self.recons_types:
            if recons_type == 'naive_no_rep' or recons_type  == 'naive':
                cluster2frame_mapping_path = Path(self.kwargs['cluster2frame_mapping_path'])
                output = eval(recons_type+'_reconstruction')(seq_names, contiguous_cluster_seqs, self.ground_truth_data, cluster2frame_mapping_path, verbose=False)
                if recons_type == 'naive_no_rep':
                    recons, faulty = output
                else:
                    recons = output
            elif recons_type == 'very_naive':
                cluster2keypoint_mapping_path = Path(self.kwargs['cluster2keypoint_mapping_path'])
                recons = eval(recons_type+'_reconstruction')(seq_names, cluster_seqs, cluster2keypoint_mapping_path, verbose=False)
            recons = [upsample(keypoint, self.recons_upsample_ratio) for keypoint in recons]
            
            for filter in self.filters:
                if filter == 'none':
                    pass
                elif filter == 'spline':
                    recons = [spline_filter1d(keypoint, axis=0) for keypoint in recons]
                elif filter == 'uniform':
                    recons = [uniform_filter1d(keypoint, size=int(self.fps/4), axis=0) for keypoint in recons]
                else :
                    raise NameError(f'No such filter {filter}')

                mpjpe_per_sequence=mpjpe3d(seq_names, recons, gt)
                getattr(self, f"MPJPE_{recons_type}_{filter}").__iadd__(np.mean(mpjpe_per_sequence))

        # jts_text, poses_text, root_text, traj_text = self.transform(jts_text, lengths)
        # jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref, lengths)

        # for i in range(len(lengths)):
        #     self.APE_root += l2_norm(root_text[i], root_ref[i], dim=1).sum()
        #     self.APE_pose += l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
        #     self.APE_traj += l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
        #     self.APE_joints += l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

        #     root_sigma_text = variance(root_text[i], lengths[i], dim=0)
        #     root_sigma_ref = variance(root_ref[i], lengths[i], dim=0)
        #     self.AVE_root += l2_norm(root_sigma_text, root_sigma_ref, dim=0)

        #     traj_sigma_text = variance(traj_text[i], lengths[i], dim=0)
        #     traj_sigma_ref = variance(traj_ref[i], lengths[i], dim=0)
        #     self.AVE_traj += l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

        #     poses_sigma_text = variance(poses_text[i], lengths[i], dim=0)
        #     poses_sigma_ref = variance(poses_ref[i], lengths[i], dim=0)
        #     self.AVE_pose += l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

        #     jts_sigma_text = variance(jts_text[i], lengths[i], dim=0)
        #     jts_sigma_ref = variance(jts_ref[i], lengths[i], dim=0)
        #     self.AVE_joints += l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)

    # def transform(self, joints: Tensor, lengths):
    #     features = self.rifke(joints)

    #     ret = self.rifke.extract(features)
    #     root_y, poses_features, vel_angles, vel_trajectory_local = ret

    #     # already have the good dimensionality
    #     angles = torch.cumsum(vel_angles, dim=-1)
    #     # First frame should be 0, but if infered it is better to ensure it
    #     angles = angles - angles[..., [0]]

    #     cos, sin = torch.cos(angles), torch.sin(angles)
    #     rotations = matrix_of_angles(cos, sin, inv=False)

    #     # Get back the local poses
    #     poses_local = rearrange(poses_features, "... (joints xyz) -> ... joints xyz", xyz=3)

    #     # Rotate the poses
    #     poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]], rotations)
    #     poses = torch.stack((poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

    #     # Rotate the vel_trajectory
    #     vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local, rotations)
    #     # Integrate the trajectory
    #     # Already have the good dimensionality
    #     trajectory = torch.cumsum(vel_trajectory, dim=-2)
    #     # First frame should be 0, but if infered it is better to ensure it
    #     trajectory = trajectory - trajectory[..., [0], :]

    #     # get the root joint
    #     root = torch.cat((trajectory[..., :, [0]],
    #                       root_y[..., None],
    #                       trajectory[..., :, [1]]), dim=-1)

    #     # Add the root joints (which is still zero)
    #     poses = torch.cat((0 * poses[..., [0], :], poses), -2)
    #     # put back the root joint y
    #     poses[..., 0, 1] = root_y

    #     # Add the trajectory globally
    #     poses[..., [0, 2]] += trajectory[..., None, :]

    #     if self.force_in_meter:
    #         # return results in meters
    #         return (remove_padding(poses / 1000, lengths),
    #                 remove_padding(poses_local / 1000, lengths),
    #                 remove_padding(root / 1000, lengths),
    #                 remove_padding(trajectory / 1000, lengths))
    #     else:
    #         return (remove_padding(poses, lengths),
    #                 remove_padding(poses_local, lengths),
    #                 remove_padding(root, lengths),
    #                 remove_padding(trajectory, lengths))

if __name__ == '__main__':
    pdb.set_trace()
