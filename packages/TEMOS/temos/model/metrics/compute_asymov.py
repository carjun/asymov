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
sys.path.append(str(Path(__file__).resolve().parents[5]))
from viz_utils import add_traj, mpjpe3d, change_fps, very_naive_reconstruction, naive_reconstruction, naive_no_rep_reconstruction
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
    def __init__(self, traj: bool, recons_types: List[str], filters: List[str], gt_path: str,
                 recons_fps: float, pred_fps: float, gt_fps: float, num_mw_clusters: int,
                 decoding_scheme: str, beam_width: int,
                #  jointstype: str = "mmm",
                #  force_in_meter: bool = True,
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.traj=traj
        self.recons_types = recons_types
        self.filters = filters
        self.recons_fps = recons_fps
        self.gt_fps = gt_fps
        self.pred_fps = pred_fps
        self.num_clusters = num_mw_clusters
        self.decoding_scheme = decoding_scheme
        if decoding_scheme == 'greedy':
            beam_width = 1
        self.beam_width = beam_width
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

        self.MPJPE_metrics=[]
        self.APE_metrics=[]
        self.AVE_metrics=[]
        for recons_type in self.recons_types:
            for filter in self.filters:
                # APE
                self.add_state(f"APE_root_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"APE_traj_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                # self.add_state(f"APE_pose_{recons_type}_{filter}", default=torch.zeros(20), dist_reduce_fx="sum")
                self.add_state(f"APE_joints_{recons_type}_{filter}", default=torch.zeros(21), dist_reduce_fx="sum")
                self.APE_metrics.extend([f"APE_{i}_{recons_type}_{filter}" for i in ["root", "traj", 
                                                                                    #  "pose", 
                                                                                     "joints"]])

                # AVE
                self.add_state(f"AVE_root_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"AVE_traj_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                # self.add_state(f"AVE_pose_{recons_type}_{filter}", default=torch.zeros(20), dist_reduce_fx="sum")
                self.add_state(f"AVE_joints_{recons_type}_{filter}", default=torch.zeros(21), dist_reduce_fx="sum")
                self.AVE_metrics.extend([f"AVE_{i}_{recons_type}_{filter}" for i in ["root", "traj", 
                                                                                    #  "pose", 
                                                                                     "joints"]])
                
                # MPJPE
                self.add_state(f"MPJPE_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.MPJPE_metrics.append(f"MPJPE_{recons_type}_{filter}")

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics + self.MPJPE_metrics

    def compute(self):
        count = self.count_good
        APE_metrics = {metric: getattr(self, metric) / count for metric in self.APE_metrics}
        
        count_seq = self.count_good_seq
        AVE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.AVE_metrics}

        for recons_type in self.recons_types:
            for filter in self.filters:
                # Compute average of APEs
                # APE_metrics[f"APE_mean_pose_{recons_type}_{filter}"] = getattr(self, f"APE_pose_{recons_type}_{filter}").mean() / count
                APE_metrics[f"APE_mean_joints_{recons_type}_{filter}"] = getattr(self, f"APE_joints_{recons_type}_{filter}").mean() / count
                # Compute average of AVEs
                # AVE_metrics[f"AVE_mean_pose_{recons_type}_{filter}"] = getattr(self, f"AVE_pose_{recons_type}_{filter}").mean() / count_seq
                AVE_metrics[f"AVE_mean_joints_{recons_type}_{filter}"] = getattr(self, f"AVE_joints_{recons_type}_{filter}").mean() / count_seq

                # Remove arrays
                # APE_metrics.pop(f"APE_pose_{recons_type}_{filter}")
                APE_metrics.pop(f"APE_joints_{recons_type}_{filter}")
                # AVE_metrics.pop(f"AVE_pose_{recons_type}_{filter}")
                AVE_metrics.pop(f"AVE_joints_{recons_type}_{filter}")

        # Compute average of MPJPEs
        MPJPE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.MPJPE_metrics}

        return {**APE_metrics, **AVE_metrics, **MPJPE_metrics,
                'all_seq':self.count_seq, 'good_seq':self.count_good_seq, 'good_seq_%': self.count_good_seq/self.count_seq,
                'all_pred':self.count, 'good_pred':self.count_good, 'good_pred_%': self.count_good/self.count
                }

    def update(self, seq_names: List[str], cluster_seqs: List[Tensor], traj: List[Tensor] = None):
        if self.traj:
            assert traj is not None

        seq_names_with_beams = []
        for i in range(self.beam_width):
            seq_names_with_beams.extend([f"{seq_name}_{i}" for seq_name in seq_names])
        assert len(seq_names)==(len(cluster_seqs)/self.beam_width)
        self.count_seq += len(seq_names)
        self.count += sum([cluster_seq.shape[0] for cluster_seq in cluster_seqs])
        good_idx = [i for i,cluster_seq in enumerate(cluster_seqs) \
            if cluster_seq.max()<self.num_clusters and cluster_seq.min()>=0]
        
        # now that we have beams, removing bad ones will be confusing while aggregation
        assert good_idx == list(range(self.beam_width*self.count_seq))
        
        # get good sequences (no <unk> or <pad>)
        # seq_names = [seq_names[i] for i in good_idx]
        # cluster_seqs = [cluster_seqs[i] for i in good_idx]
        self.count_good_seq += len(seq_names)
        self.count_good += sum([cluster_seq.shape[0] for cluster_seq in cluster_seqs])

        # get GT
        gt = [self.ground_truth_data[name][:5000, :, :] for name in seq_names]
        gt = [change_fps(keypoint, self.gt_fps, self.recons_fps) for keypoint in gt]
        gt_with_beams = gt*self.beam_width

        # get contiguous cluster sequences (grouping contiguous identical clusters)
        cluster_seqs = [cluster_seq.cpu().numpy() for cluster_seq in cluster_seqs]
        if ('naive_no_rep' in self.recons_types) or ('naive' in self.recons_types):
            contiguous_frame2cluster_mapping = {"name":[], "idx":[], "cluster":[], "length":[]}
            for name, cluster_seq in zip(seq_names_with_beams, cluster_seqs):
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
            contiguous_cluster_seqs = [contiguous_frame2cluster_mapping[contiguous_frame2cluster_mapping['name']==name][['cluster', 'length']].reset_index(drop=True) for name in seq_names_with_beams]

        # reconstruct from predicted clusters using different strategies
        for recons_type in self.recons_types:
            if recons_type == 'naive_no_rep' or recons_type  == 'naive':
                cluster2frame_mapping_path = Path(self.kwargs['cluster2frame_mapping_path'])
                output = eval(recons_type+'_reconstruction')(seq_names_with_beams, contiguous_cluster_seqs, self.ground_truth_data, cluster2frame_mapping_path, verbose=False)
                if recons_type == 'naive_no_rep':
                    recons, faulty = output
                else:
                    recons = output
            elif recons_type == 'very_naive':
                cluster2keypoint_mapping_path = Path(self.kwargs['cluster2keypoint_mapping_path'])
                recons = eval(recons_type+'_reconstruction')(seq_names_with_beams, cluster_seqs, cluster2keypoint_mapping_path, verbose=False)
            # traj inclusion
            if self.traj:
                recons = add_traj(recons, traj)
            recons = [change_fps(keypoint, self.pred_fps, self.recons_fps) for keypoint in recons]

            # apply different filters
            for filter in self.filters:
                if filter == 'none':
                    pass
                elif filter == 'spline':
                    recons = [spline_filter1d(keypoint, axis=0) for keypoint in recons]
                elif filter == 'uniform':
                    recons = [uniform_filter1d(keypoint, size=int(self.recons_fps/4), axis=0) for keypoint in recons]
                else :
                    raise NameError(f'No such filter {filter}')

                # MPJPE
                mpjpe_with_beams=mpjpe3d(seq_names_with_beams, recons, gt_with_beams)
                mpjpe_per_sequence=[np.mean(mpjpe_with_beams[i::self.count_good_seq]) for i in range(self.count_good_seq)]
                getattr(self, f"MPJPE_{recons_type}_{filter}").__iadd__(np.mean(mpjpe_per_sequence))

                # AVE and APE
                lengths = [min(recons[i].shape[0], gt_with_beams[i].shape[0]) for i in range(len(recons))]

                #TODO : TEMOS transforms to get jts, root, poses and traj
                
                # jts_text, poses_text, root_text, traj_text = self.transform(jts_text, lengths)
                jts_text = [torch.from_numpy(keypoint)[:l] for keypoint, l in zip(recons, lengths)]
                # poses_text = jts_text
                root_text = [jts[..., 0, :] for jts in jts_text]
                traj_text = [jts[..., 0, [0, 2]] for jts in jts_text]

                # jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref, lengths)
                jts_ref = [torch.from_numpy(keypoint)[:l] for keypoint, l in zip(gt_with_beams, lengths)]
                # poses_ref = jts_ref
                root_ref = [jts[..., 0, :] for jts in jts_ref]
                traj_ref = [jts[..., 0, [0, 2]] for jts in jts_ref]

                for seq in range(self.count_good_seq): #aggregate over beams and update
                    APE_root_per_beam =  torch.stack([l2_norm(root_text[i], root_ref[i], dim=1).sum() for i in range(seq, len(recons), self.count_good_seq)])
                    APE_root = APE_root_per_beam.mean(0)
                    getattr(self, f"APE_root_{recons_type}_{filter}").__iadd__(APE_root)
                    APE_traj_per_beam = torch.stack([l2_norm(traj_text[i], traj_ref[i], dim=1).sum() for i in range(seq, len(recons), self.count_good_seq)])
                    APE_traj = APE_traj_per_beam.mean(0)
                    getattr(self, f"APE_traj_{recons_type}_{filter}").__iadd__(APE_traj)
                    # APE_pose_per_beam = torch.stack([l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0) for i in range(seq, len(recons), self.count_good_seq)])
                    # APE_pose = APE_pose_per_beam.mean(0)
                    # getattr(self, f"APE_pose_{recons_type}_{filter}").__iadd__(APE_pose)
                    APE_joints_per_beam = torch.stack([l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0) for i in range(seq, len(recons), self.count_good_seq)])
                    APE_joints = APE_joints_per_beam.mean(0)
                    getattr(self, f"APE_joints_{recons_type}_{filter}").__iadd__(APE_joints)

                    root_sigma_text = [variance(root_text[i], lengths[i], dim=0) for i in range(seq, len(recons), self.count_good_seq)]
                    root_sigma_ref = [variance(root_ref[i], lengths[i], dim=0) for i in range(seq, len(recons), self.count_good_seq)]
                    AVE_root_per_beam = torch.stack([l2_norm(i, j, dim=0) for i,j in zip(root_sigma_text, root_sigma_ref)])
                    AVE_root = AVE_root_per_beam.mean(0)
                    getattr(self, f"AVE_root_{recons_type}_{filter}").__iadd__(AVE_root)

                    traj_sigma_text = [variance(traj_text[i], lengths[i], dim=0) for i in range(seq, len(recons), self.count_good_seq)]
                    traj_sigma_ref = [variance(traj_ref[i], lengths[i], dim=0) for i in range(seq, len(recons), self.count_good_seq)]
                    AVE_traj_per_beam = torch.stack([l2_norm(i, j, dim=0) for i,j in zip(traj_sigma_text, traj_sigma_ref)])
                    AVE_traj = AVE_traj_per_beam.mean(0)
                    getattr(self, f"AVE_traj_{recons_type}_{filter}").__iadd__(AVE_traj)

                    # poses_sigma_text = [variance(poses_text[i], lengths[i], dim=0) for i in range(seq, len(recons), self.count_good_seq)]
                    # poses_sigma_ref = [variance(poses_ref[i], lengths[i], dim=0) for i in range(seq, len(recons), self.count_good_seq)]
                    # AVE_pose_per_beam = torch.stack([l2_norm(i, j, dim=1) for i,j in zip(poses_sigma_text, poses_sigma_ref)])
                    # AVE_pose = AVE_pose_per_beam.mean(0)
                    # getattr(self, f"AVE_pose_{recons_type}_{filter}").__iadd__(AVE_pose)

                    jts_sigma_text = [variance(jts_text[i], lengths[i], dim=0) for i in range(seq, len(recons), self.count_good_seq)]
                    jts_sigma_ref = [variance(jts_ref[i], lengths[i], dim=0) for i in range(seq, len(recons), self.count_good_seq)]
                    AVE_joints_per_beam = torch.stack([l2_norm(i, j, dim=1) for i,j in zip(jts_sigma_text, jts_sigma_ref)])
                    AVE_joints = AVE_joints_per_beam.mean(0)
                    getattr(self, f"AVE_joints_{recons_type}_{filter}").__iadd__(AVE_joints)
