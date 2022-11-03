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
sys.path.append(str(Path(__file__).parents[5]))
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
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_good", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_good_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        # APE
        self.add_state("APE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.add_state("APE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        self.add_state("APE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.APE_metrics = ["APE_root", "APE_traj",
                            # "APE_pose",
                            "APE_joints"]

        # AVE
        self.add_state("AVE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.add_state("AVE_pose", default=torch.zeros(20), dist_reduce_fx="sum")
        self.add_state("AVE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.AVE_metrics = ["AVE_root", "AVE_traj",
                            # "AVE_pose",
                            "AVE_joints"]

        # MPJPE
        self.MPJPE_metrics=[]
        for recons_type in self.recons_types:
            for filter in self.filters:
                self.add_state(f"MPJPE_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.MPJPE_metrics.append(f"MPJPE_{recons_type}_{filter}")

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics + self.MPJPE_metrics

    def compute(self):
        count = self.count_good
        APE_metrics = {metric: getattr(self, metric) / count for metric in self.APE_metrics}

        # Compute average of APEs
        # APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count

        # Remove arrays
        # APE_metrics.pop("APE_pose")
        APE_metrics.pop("APE_joints")

        count_seq = self.count_good_seq
        AVE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.AVE_metrics}

        # Compute average of AVEs
        # AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq

        # Remove arrays
        # AVE_metrics.pop("AVE_pose")
        AVE_metrics.pop("AVE_joints")

        # Compute average of MPJPEs
        MPJPE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.MPJPE_metrics}

        return {**APE_metrics, **AVE_metrics, **MPJPE_metrics,
                'all_seq':self.count_seq, 'good_seq':self.count_good_seq, 'good_seq_%': self.count_good_seq/self.count_seq,
                'all_pred':self.count, 'good_pred':self.count_good, 'good_pred_%': self.count_good/self.count
                }

    def update(self, seq_names: List[str], cluster_seqs: List[Tensor]):
        assert len(seq_names)==len(cluster_seqs)
        self.count_seq += len(seq_names)
        self.count += sum([cluster_seq.shape[0] for cluster_seq in cluster_seqs])
        good_idx = [i for i,cluster_seq in enumerate(cluster_seqs) \
            if cluster_seq.max()<self.num_clusters and cluster_seq.min()>=0]

        # get good sequences (no <unk> or <pad>)
        seq_names = [seq_names[i] for i in good_idx]
        cluster_seqs = [cluster_seqs[i] for i in good_idx]
        self.count_good_seq += len(seq_names)
        self.count_good += sum([cluster_seq.shape[0] for cluster_seq in cluster_seqs])

        # get GT
        gt = [self.ground_truth_data[name][:5000, :, :] for name in seq_names]
        gt = [downsample(keypoint, self.gt_downsample_ratio) for keypoint in gt]

        # get contiguous cluster sequences (grouping contiguous identical clusters)
        cluster_seqs = [cluster_seq.cpu().numpy() for cluster_seq in cluster_seqs]
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

        # reconstruct from predicted clusters using different strategies
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

            # apply different filters
            for filter in self.filters:
                if filter == 'none':
                    pass
                elif filter == 'spline':
                    recons = [spline_filter1d(keypoint, axis=0) for keypoint in recons]
                elif filter == 'uniform':
                    recons = [uniform_filter1d(keypoint, size=int(self.fps/4), axis=0) for keypoint in recons]
                else :
                    raise NameError(f'No such filter {filter}')

                # MPJPE
                mpjpe_per_sequence=mpjpe3d(seq_names, recons, gt)
                getattr(self, f"MPJPE_{recons_type}_{filter}").__iadd__(np.mean(mpjpe_per_sequence))

                # AVE and APE
                lengths = [min(recons[i].shape[0], gt[i].shape[0]) for i in range(len(recons))]

                # jts_text, poses_text, root_text, traj_text = self.transform(jts_text, lengths)
                jts_text = [torch.from_numpy(keypoint)[:l] for keypoint, l in zip(recons, lengths)]
                # poses_text = jts_text
                root_text = [jts[..., 0, :] for jts in jts_text]
                traj_text = [jts[..., 0, [0, 2]] for jts in jts_text]

                # jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref, lengths)
                jts_ref = [torch.from_numpy(keypoint)[:l] for keypoint, l in zip(gt, lengths)]
                # poses_ref = jts_ref
                root_ref = [jts[..., 0, :] for jts in jts_ref]
                traj_ref = [jts[..., 0, [0, 2]] for jts in jts_ref]

                for i in range(len(recons)):
                    self.APE_root += l2_norm(root_text[i], root_ref[i], dim=1).sum()
                    # self.APE_pose += l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
                    self.APE_traj += l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
                    self.APE_joints += l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

                    root_sigma_text = variance(root_text[i], lengths[i], dim=0)
                    root_sigma_ref = variance(root_ref[i], lengths[i], dim=0)
                    self.AVE_root += l2_norm(root_sigma_text, root_sigma_ref, dim=0)

                    traj_sigma_text = variance(traj_text[i], lengths[i], dim=0)
                    traj_sigma_ref = variance(traj_ref[i], lengths[i], dim=0)
                    self.AVE_traj += l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

                #     poses_sigma_text = variance(poses_text[i], lengths[i], dim=0)
                #     poses_sigma_ref = variance(poses_ref[i], lengths[i], dim=0)
                #     self.AVE_pose += l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

                    jts_sigma_text = variance(jts_text[i], lengths[i], dim=0)
                    jts_sigma_ref = variance(jts_ref[i], lengths[i], dim=0)
                    self.AVE_joints += l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)
