import json
import os
from glob import glob
from typing import Dict, Optional, Callable, Union
from omegaconf import ListConfig
import logging
import pickle
from functools import partial

import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from pathlib import Path

from temos.tools.easyconvert import matrix_to, axis_angle_to
# from temos.transforms import Transform
from temos.data.sampling import subsample
from temos.data.tools.smpl import smpl_data_to_matrix_and_trans
from temos.data.tools.collate import * #tokenizer and collate functions

from .base import BASEDataModule
from .utils import get_split_keyids#, mt_terminal_transform, mt_mwid_transform, sequential_transforms

logger = logging.getLogger(__name__)

class KITMotionWordMTDataModule(BASEDataModule):
    def __init__(self, 
                 special_symbols: Union[List[str],ListConfig] = ['<pad>', '<bos>', '<eos>', '<unk>'],
                 num_mw_clusters: int = 1000,
                 span: bool = True,
                 traj: bool = True,
                 batch_size: int = 32,
                 num_workers: int = 16,
                 collate_fn: Callable = collate_motion_words_and_text_mt,
                 **kwargs):
        self.save_hyperparameters()
        
        self.Dataset = KITMotionWord
        # self.text_token_transform = get_tokenizer('spacy', language='en_core_web_sm')

        # pdb.set_trace()
        self.PAD_IDX, self.BOS_IDX, self.EOS_IDX, self.UNK_IDX = \
            special_symbols.index('<pad>'), special_symbols.index('<bos>'), special_symbols.index('<eos>'), special_symbols.index('<unk>')
        self.text_special_symbols = list(special_symbols)
        
        train_data = [ann for annotations in self.train_dataset.texts_data.values() for ann in annotations]
        txt_tokens = [tokenizer(txt) for txt in train_data]
        self.text_vocab = build_vocab_from_iterator(txt_tokens, min_freq=1, specials=self.text_special_symbols, special_first=True)
        assert [self.PAD_IDX, self.BOS_IDX, self.EOS_IDX, self.UNK_IDX] == self.text_vocab(self.text_special_symbols)
        self.text_vocab.set_default_index(self.UNK_IDX)
        
        self.mw_special_symbols = list(special_symbols)
        # self.mw_shift_transform = partial(mt_mwid_transform, num_special = len(self.mw_special_symbols))
        
        # self.terminal_transform = partial(mt_terminal_transform, BOS_IDX=self.BOS_IDX, EOS_IDX=self.EOS_IDX)
        # self.text_transform = sequential_transforms(self.text_token_transform, #Tokenization
        #                                             self.text_vocab_transform, #Numericalization
        #                                             self.terminal_transform) # Add BOS/EOS and create tensor
        # self.text_transform = [self.text_token_transform, #Tokenization
        #                        self.text_vocab_transform, #Numericalization
        #                        self.terminal_transform] # Add BOS/EOS and create tensor
        # self.mw_transform = sequential_transforms(self.mw_shift_transform, #Numericalization
        #                                           self.terminal_transform) # Add BOS/EOS and create tensor
        # self.mw_transform = [self.mw_shift_transform, #Numericalization
        #                     self.terminal_transform] # Add BOS/EOS and create tensor
        
        self.text_vocab_size = self.text_vocab.__len__()
        print(f"text_vocab_size set to : {self.text_vocab_size}")
        self.mw_vocab_size = num_mw_clusters + len(self.mw_special_symbols)
        print(f"mw_vocab_size set to : {self.mw_vocab_size}")
        self.max_frames = max(self.train_dataset.durations.values())
        print(f"max_frames set to : {self.max_frames}")
        
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=partial(collate_fn, text_vocab=self.text_vocab, special_symbols=list(special_symbols), traj=traj, span=span))

class KITMotionWordDataModule(BASEDataModule):
    def __init__(self, data_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 vocab_size: int = 1000,
                 collate_fn: Callable = collate_motion_words_and_text,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=partial(collate_fn, vocab_size=vocab_size))
        self.save_hyperparameters()
        self.Dataset = KITMotionWord

        # sample_overrides = {"split": "train", "tiny": True,
        #                     "progress_bar": False}
        # self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        # self.vocab_size = vocab_size
        # self.dataloader_options.update({'vocab_size': self.vocab_size})
        # self.transforms = self._sample_set.transforms



class KITMotionWord(Dataset):
    dataname = "KIT Motion-Language Motion Word"

    def __init__(self, datapath: str, mw_dataname: str, 
                 traj: bool, traj_dataname: str, 
                 span: bool, span_dataname: str,
                 ann_dataname: str,
                 splitpath: str,
                 vocab_size: int,
                #  transforms: Transform,
                 split: str = "train",
                #  transforms_xyz: Optional[Transform] = None,
                #  transforms_smpl: Optional[Transform] = None,
                #  correspondance_path: str = None,
                #  amass_path: str = None,
                #  smplh_path: str = None,
                 sampler=None,
                 framerate: float = 12.5,
                 progress_bar: bool = True,
                 pick_one_text: bool = True,
                #  load_amass_data=False,
                #  load_with_rot=False,
                 downsample=True,
                 tiny: bool = False, **kwargs):

        self.split = split
        # self.load_amass_data = load_amass_data
        # self.load_with_rot = load_with_rot
        self.downsample = downsample
        self.traj = traj
        self.span = span

        # if load_amass_data and not self.load_with_rot:
        #     self.transforms_xyz = transforms_xyz
        #     self.transforms_smpl = transforms_smpl
        #     self.transforms = transforms_xyz
        # else:
        # self.transforms = transforms

        self.sampler = sampler
        self.pick_one_text = pick_one_text

        super().__init__()
        keyids = get_split_keyids(path=splitpath, split=split)

        mw_data = {}
        texts_data = {}
        durations = {}

        # if load_amass_data:
        #     with open(correspondance_path) as correspondance_path_file:
        #         kitml_correspondances = json.load(correspondance_path_file)

        if tiny:
            keyids = keyids[:2]

        if progress_bar:
            enumerator = enumerate(tqdm(keyids, f"Loading KIT motion word {split}"))
        else:
            enumerator = enumerate(keyids)


        datapath = Path(datapath)

        num_bad = 0
        # if load_amass_data:
        #     bad_smpl = 0
        #     good_smpl = 0
        
        with open(datapath/mw_dataname, 'rb') as f:
            motion_words_data = pickle.load(f)
        
        if self.span:
            span_data = {}
            with open(datapath/span_dataname, 'rb') as f:
                span = pickle.load(f)
        if self.traj:
            traj_data = {}
            with open(datapath/traj_dataname, 'rb') as f:
                trajectory_data = pickle.load(f)

        with open(datapath/ann_dataname, 'r') as f:
            ann_data = json.load(f)["anns"]
        
        for i, keyid in enumerator:

            text = ann_data[keyid]
            # ann_data, success = load_annotation(keyid, datapath)
            # if not success:
            #     logger.error(f"{keyid} has no annotations")
            #     continue

            # read smpl params
            # if load_amass_data:
            #     smpl_data, success = load_amass_keyid(keyid, amass_path,
            #                                           correspondances=kitml_correspondances)

            #     if not success:
            #         bad_smpl += 1
            #         continue
            #     else:
            #         good_smpl += 1

            #     smpl_data, duration = downsample_amass(smpl_data, downsample=self.downsample, framerate=framerate)
            #     smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True)
            # read xyz joints in MMM format
            # else:
            # joints = load_mmm_keyid(keyid, datapath)

            # TODO : set argument for downsample and upsample in required places
            motion_words, duration = downsample_motion_words(motion_words_data[keyid], downsample=self.downsample, framerate=framerate)

            if split != "test" and not tiny:
                # Accept or not the sample, based on the duration
                if not self.sampler.accept(duration):
                    num_bad += 1
                    continue

            # Load rotation features (rfeats) data from AMASS
            # if load_amass_data and load_with_rot:
            #     features = self.transforms.rots2rfeats(smpl_data)
            # # Load xyz features (jfeats) data from AMASS
            # elif load_amass_data and not load_with_rot:
            #     joints = self.transforms_smpl.rots2joints(smpl_data)
            #     features = self.transforms_xyz.joints2jfeats(joints)
            # Load xyz features (jfeats) data from MMM
            # else:
            # features = self.transforms.joints2jfeats(joints)

            mw_data[keyid] = motion_words
            texts_data[keyid] = text
            durations[keyid] = duration

            if self.span:
                spans = span[keyid]
                assert len(motion_words) == len(spans)
                span_data[keyid] = spans
            if self.traj:
                residual_traj, _ = residual_downsample_traj(trajectory_data[keyid], downsample=self.downsample, framerate=framerate)
                assert len(residual_traj) == duration
                traj_data[keyid] = residual_traj

        # if load_amass_data and not tiny:
        #     percentage = 100 * bad_smpl / (bad_smpl + good_smpl)
        #     logger.info(f"There are {bad_smpl} sequences not found ({percentage:.4}%) in AMASS.")

        if split != "test" and not tiny:
            total = len(mw_data)
            percentage = 100 * num_bad / (total+num_bad)
            logger.info(f"There are {num_bad} sequences rejected by the sampler ({percentage:.4}%).")

        self.mw_data = mw_data
        self.texts_data = texts_data
        if self.span:
            self.span_data = span_data
        if self.traj:
            self.traj_data = traj_data

        self.keyids = list(mw_data.keys())
        self._split_index = list(self.keyids)
        self.durations = durations
        self.vocab_size = vocab_size

    def _load_span(self, keyid):
        span = self.span_data[keyid]
        return span
    def _load_traj(self, keyid):#, frame_ix=None):
        traj = self.traj_data[keyid]
        # datastruct = self.transforms.Datastruct(features=features)
        return traj

    def _load_motion_words(self, keyid):#, frame_ix=None):
        motion_words = self.mw_data[keyid]
        # datastruct = self.transforms.Datastruct(features=features)
        return motion_words

    def _load_text(self, keyid):
        sequences = self.texts_data[keyid]
        if not self.pick_one_text:
            return sequences
        n = len(sequences)
        if self.split != "test":
            index = np.random.randint(n)
        else:
            # Only the first one in evaluation
            index = 0
        text = sequences[index]
        return text

    def load_keyid(self, keyid):
        num_frames = self.durations[keyid]
        # frame_ix = self.sampler(num_frames)

        motion_words = self._load_motion_words(keyid)#, frame_ix)
        text = self._load_text(keyid)
        element = {"motion_words": motion_words, "text": text,
                   "length": len(motion_words), "keyid": keyid}
        if self.span:
            span = self._load_span(keyid)
            element["span"] = span
        if self.traj:
            traj = self._load_traj(keyid)
            element["traj"]=traj
        return element

    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid)

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"


# def load_annotation(keyid, datapath):
#     metapath = datapath / (keyid + "_meta.json")
#     metadata = json.load(metapath.open())

#     if metadata["nb_annotations"] == 0:
#         logger.error(f"{keyid} has no annotations")
#         return None, False

#     annpath = datapath / (keyid + "_annotations.json")
#     ann_data = json.load(annpath.open())
#     assert len(ann_data) == metadata["nb_annotations"]
#     return ann_data, True


# def load_mmm_keyid(keyid, datapath):
#     xyzpath = datapath / (keyid + "_fke.csv")
#     xyzdata = pandas.read_csv(xyzpath, index_col=0)
#     joints = np.array(xyzdata).reshape(-1, 21, 3)
#     return joints

def downsample_motion_words(motion_words, *, downsample, framerate):
    nframes_total = len(motion_words)
    last_framerate = 100

    if downsample:
        frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
    else:
        frames = np.arange(nframes_total, dtype='int32')

    duration = len(frames)
    motion_words = torch.from_numpy(np.array(motion_words, dtype='int32')[frames])
    return motion_words, duration

def residual_downsample_traj(traj, *, downsample, framerate):
    nframes_total = len(traj)
    last_framerate = 100

    if downsample:
        frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
    else:
        frames = np.arange(nframes_total, dtype='int32')

    duration = len(frames)
    traj = traj[frames]
    residual_traj = torch.from_numpy(np.diff(traj, axis=0, prepend=traj[0:1]))
    return residual_traj, duration

# def load_amass_keyid(keyid, amass_path, *, correspondances):
#     identifier = correspondances[keyid]["identifier"]
#     smpl_keyid_path = correspondances[keyid]["path"]

#     if identifier == "kit":
#         smpl_datapath = Path(amass_path) / "KIT" / "KIT" / smpl_keyid_path
#     elif identifier == "cmu":
#         smpl_datapath = Path(amass_path) / "CMU" / "CMU" / smpl_keyid_path

#         if not os.path.exists(smpl_datapath):
#             # try with EKUT folder instead
#             smpl_datapath = Path(amass_path) / "EKUT" / "EKUT" / smpl_keyid_path

#             # File not found
#             if not os.path.exists(smpl_datapath):
#                 return None, False
#     else:
#         raise TypeError(f"{identifier} identifier not recognized.")
#     try:
#         smpl_data = np.load(smpl_datapath)
#     except FileNotFoundError:
#         return None, False

#     smpl_data = {x: smpl_data[x] for x in smpl_data.files}
#     return smpl_data, True

# def downsample_amass(smpl_data, *, downsample, framerate):
#     nframes_total = len(smpl_data["poses"])
#     last_framerate = smpl_data["mocap_framerate"].item()

#     if downsample:
#         frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
#     else:
#         frames = np.arange(nframes_total)

#     duration = len(frames)

#     # subsample
#     smpl_data = {"poses": torch.from_numpy(smpl_data["poses"][frames]).float(),
#                  "trans": torch.from_numpy(smpl_data["trans"][frames]).float()}
#     return smpl_data, duration
