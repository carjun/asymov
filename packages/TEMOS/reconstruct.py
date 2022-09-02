import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import os
import yaml, pprint, json

import sys
sys.path.append('../../')
from viz import naive_reconstruction_no_rep, naive_reconstruction, very_naive_reconstruction, ground_truth_construction
import temos.launch.prepare #noqa

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="reconstruct")
def _reconstruct(cfg: DictConfig):
    return reconstruct(cfg)

def reconstruct(cfg: DictConfig) -> None:
    # Load last config
    pred_dir = Path(hydra.utils.to_absolute_path(cfg.folder))
    data_dir = Path(cfg.data_dir)
    ground_truth_path = data_dir / "xyz_data.pkl"
    
    logger.info("Predicted data is taken from: ")
    logger.info(f"{pred_dir}")

    seq_names = sorted([i[:-4] for i in os.listdir(pred_dir) if i.endswith('.npy')])
    logger.info(f"Total sequences: {len(seq_names)}")
    recons_names = seq_names[:10]
    
    frame2cluster_mapping_dir = pred_dir
    contiguous_frame2cluster_mapping_path = pred_dir / "contiguous_frame2cluster_mapping.pkl"
    cluster2keypoint_mapping_path = data_dir / f'proxy_centers_tr_{cfg.vocab_size}.pkl'
    cluster2frame_mapping_path = data_dir /  f'proxy_centers_tr_complete_{cfg.vocab_size}.pkl'

    
    if cfg.reconstruct == True:
        recons_dir = pred_dir/'reconstructions'
        recons_dir.mkdir(exist_ok=True, parents=True)
        logger.info("Reconstructed vdeos will be stored in:")
        logger.info(f"{recons_dir}")
    else:
        recons_dir = None
        logger.info("Reconstructed videos will not be generated")
        # recons_names = []
    
    #No filter
    very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, ground_truth_path, cluster2keypoint_mapping_path, cfg.sk_type, frames_dir=recons_dir, frame2cluster_mapping_dir=frame2cluster_mapping_dir, recons_names=recons_names)
    naive_mpjpe_mean = naive_reconstruction(seq_names, ground_truth_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, cfg.sk_type, frames_dir=recons_dir, recons_names=recons_names)
    naive_no_rep_mpjpe_mean, faulty = naive_reconstruction_no_rep(seq_names, ground_truth_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, cfg.sk_type, frames_dir=recons_dir, recons_names=recons_names)


    #uniform filter
    uni_very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, ground_truth_path, cluster2keypoint_mapping_path, cfg.sk_type, filter = 'uniform', frames_dir=recons_dir, frame2cluster_mapping_dir=frame2cluster_mapping_dir, recons_names=recons_names)
    uni_naive_mpjpe_mean = naive_reconstruction(seq_names, ground_truth_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, cfg.sk_type, filter='uniform', frames_dir=recons_dir, recons_names=recons_names)
    uni_naive_no_rep_mpjpe_mean, faulty = naive_reconstruction_no_rep(seq_names, ground_truth_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, cfg.sk_type, filter='uniform', frames_dir=recons_dir, recons_names=recons_names)
    
    
    #spline filter
    spline_very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, ground_truth_path, cluster2keypoint_mapping_path, cfg.sk_type, filter = 'spline', frames_dir=recons_dir, frame2cluster_mapping_dir=frame2cluster_mapping_dir, recons_names=recons_names)
    spline_naive_mpjpe_mean = naive_reconstruction(seq_names, ground_truth_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, cfg.sk_type, filter='spline', frames_dir=recons_dir, recons_names=recons_names)
    spline_naive_no_rep_mpjpe_mean, faulty = naive_reconstruction_no_rep(seq_names, ground_truth_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, cfg.sk_type, filter='spline', frames_dir=recons_dir, recons_names=recons_names)
    
    if cfg.reconstruct_ground:
        #original video
        ground_truth_construction(recons_names, ground_truth_path, cfg.sk_type, frames_dir= data_dir / 'constructions')
    
    print('very naive mpjpe : ', very_naive_mpjpe_mean)
    print('naive mpjpe : ', naive_mpjpe_mean)
    print('naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)
    print(f'{len(faulty)} faulty seqs : {faulty}')
    print('----------------------------------------------------')
    print('uniform filtered very naive mpjpe : ', uni_very_naive_mpjpe_mean)
    print('uniform filtered naive mpjpe : ', uni_naive_mpjpe_mean)
    print('uniform filtered naive (no rep) mpjpe : ', uni_naive_no_rep_mpjpe_mean)
    print(f'{len(faulty)} faulty seqs : {faulty}')
    print('----------------------------------------------------')
    print('spline filtered very naive mpjpe : ', spline_very_naive_mpjpe_mean)
    print('spline filtered naive mpjpe : ', spline_naive_mpjpe_mean)
    print('spline filtered naive (no rep) mpjpe : ', spline_naive_no_rep_mpjpe_mean)
    print(f'{len(faulty)} faulty seqs : {faulty}')
    
    mpjpe_table = { 'filter': ['none', 'uniform', 'spline'], 
                    'very_naive':[very_naive_mpjpe_mean, uni_very_naive_mpjpe_mean, spline_very_naive_mpjpe_mean],
                    'naive':[naive_mpjpe_mean, uni_naive_mpjpe_mean, spline_naive_mpjpe_mean],
                    'naive_no_rep':[naive_no_rep_mpjpe_mean, uni_naive_no_rep_mpjpe_mean, spline_naive_no_rep_mpjpe_mean]
                    }
    
    pd.DataFrame.from_dict(mpjpe_table).to_csv(pred_dir / "mpjpe_scores.csv")
if __name__ == '__main__':
    _reconstruct()
