from typing import List, Optional

import torch
from torch import nn
import pdb

from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from temos.model.utils.tools import remove_padding_asymov, remove_padding

from temos.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection, Accuracy, BLEUScore, SumMetric
from temos.model.base import BaseModel
from torch.distributions.distribution import Distribution


class Asymov(BaseModel):
    def __init__(self, textencoder: DictConfig,
                 motionencoder: DictConfig,
                 motiondecoder: DictConfig,
                 losses: DictConfig,
                 optim: DictConfig,
                #  transforms: DictConfig,
                 vocab_size: int,
                 vae: bool,
                 latent_dim: int,
                 **kwargs):
        super().__init__()

        self.textencoder = instantiate(textencoder)
        self.motionencoder = instantiate(motionencoder)

        # self.transforms = instantiate(transforms)
        # self.Datastruct = self.transforms.Datastruct

        self.motiondecoder = instantiate(motiondecoder)
        self.optimizer = instantiate(optim, params=self.parameters())

        self._losses = MetricCollection({split: instantiate(losses, vae=vae,
                                                            _recursive_=False)
                                         for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}

        # self.metrics = ComputeMetrics()
        #TODO: can refactor
        self.train_metrics = {'acc_mw2mw': Accuracy(num_classes=vocab_size+1, mdmc_average='samplewise',
                                              ignore_index=vocab_size, multiclass=True,# subset_accuracy=True
                                              ),
                              'acc_text2mw': Accuracy(num_classes=vocab_size+1, mdmc_average='samplewise',
                                              ignore_index=vocab_size, multiclass=True,# subset_accuracy=True
                                              ),
                              'bleu_mw2mw': BLEUScore(),
                              'bleu_text2mw': BLEUScore(),
                              'ppl_mw2mw': SumMetric(),
                              'ppl_text2mw': SumMetric(),
                             }
        self.val_metrics = {'acc_mw2mw': Accuracy(num_classes=vocab_size+1, mdmc_average='samplewise',
                                              ignore_index=vocab_size, multiclass=True,# subset_accuracy=True
                                              ),
                              'acc_text2mw': Accuracy(num_classes=vocab_size+1, mdmc_average='samplewise',
                                              ignore_index=vocab_size, multiclass=True,# subset_accuracy=True
                                              ),
                              'bleu_mw2mw': BLEUScore(),
                              'bleu_text2mw': BLEUScore(),
                              'ppl_mw2mw': SumMetric(),
                              'ppl_text2mw': SumMetric(),
                             }
        self.metrics={'train':self.train_metrics, 'val':self.val_metrics}

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None

        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict) -> List[Tensor]:
        probs_from_text = self.text_to_motion_forward(batch["text"],
                                                           batch["length"])

        return remove_padding_asymov(probs_from_text, batch["length"])

    def sample_from_distribution(self, distribution: Distribution, *,
                                 fact: Optional[bool] = None,
                                 sample_mean: Optional[bool] = False) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return distribution.loc

        # Reparameterization trick
        if fact is None:
            return distribution.rsample()

        # Resclale the eps
        eps = distribution.rsample() - distribution.loc
        latent_vector = distribution.loc + fact * eps
        return latent_vector

    def text_to_motion_forward(self, text_sentences: List[str], lengths: List[int], *,
                               return_latent: bool = False):
        # Encode the text to the latent space
        if self.hparams.vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)

        # Decode the latent vector to a motion
        probs = self.motiondecoder(latent_vector, lengths)
        # datastruct = self.Datastruct(features=features)

        if not return_latent:
            return probs
        return probs, latent_vector, distribution

    def motion_to_motion_forward(self, motion_words,
                                 lengths: Optional[List[int]] = None,
                                 return_latent: bool = False
                                 ):
        # Make sure it is on the good device
        # datastruct.transforms = self.transforms

        # Encode the motion to the latent space
        if self.hparams.vae:
            distribution = self.motionencoder(motion_words, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector: Tensor = self.motionencoder(motion_words, lengths)

        # Decode the latent vector to a motion
        probs = self.motiondecoder(latent_vector, lengths)
        # datastruct = self.Datastruct(features=features)

        if not return_latent:
            return probs
        return probs, latent_vector, distribution

    def allsplit_step(self, split: str, batch, batch_idx):
        # Encode the text/decode to a motion
        ret = self.text_to_motion_forward(batch["text"],
                                          batch["length"],
                                          return_latent=True)
        probs_from_text, latent_from_text, distribution_from_text = ret
        #[Batch size, Classes, Frames]

        # Encode the motion/decode to a motion
        ret = self.motion_to_motion_forward(batch["motion_words"],
                                            batch["length"],
                                            return_latent=True)
        probs_from_motion, latent_from_motion, distribution_from_motion = ret
        #[Batch size, Classes, Frames]

        # GT data
        motion_word_ref = batch["motion_words"].long() #long tensor for cross entropy target
        #[Batch size, Frames]

        # Compare to a Normal distribution
        if self.hparams.vae:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(distribution_from_text.loc)
            scale_ref = torch.ones_like(distribution_from_text.scale)
            distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            distribution_ref = None

        # Compute the losses
        loss = self.losses[split].update(ds_text=probs_from_text,
                                         ds_motion=probs_from_motion,
                                         ds_ref=motion_word_ref,
                                         lat_text=latent_from_text,
                                         lat_motion=latent_from_motion,
                                         dis_text=distribution_from_text,
                                         dis_motion=distribution_from_motion,
                                         dis_ref=distribution_ref)

        if loss is None:
            raise ValueError("Loss is None, this happend with torchmetrics > 0.7")

        ### Compute the metrics
        preds_from_text = probs_from_text.detach().softmax(dim=1)
        preds_from_motion = probs_from_motion.detach().softmax(dim=1)
        bs, _, frames = preds_from_text.shape
        # assert bs == preds_from_motion.shape[0] and frames == preds_from_motion.shape[2], 'train and val predictions shape mismatch'
        target = motion_word_ref.detach()

        # adding padding class to preds for compatibility with padded target motion words for Accuracy
        preds_from_text = torch.cat((preds_from_text, preds_from_text.new_zeros(bs, 1, frames)), dim=1)
        preds_from_motion = torch.cat((preds_from_motion, preds_from_motion.new_zeros(bs, 1, frames)), dim=1)

        self.metrics[split]['acc_text2mw'].update(preds_from_text, target)
        self.metrics[split]['acc_mw2mw'].update(preds_from_motion, target)

        # predicted, target motion word ids without padding for BLEU
        pred_mw_from_text = remove_padding(torch.argmax(preds_from_text, dim=1).int(), batch["length"])
        pred_mw_from_motion = remove_padding(torch.argmax(preds_from_motion, dim=1).int(), batch["length"])
        pred_mw_sents_from_text = [" ".join(map(str, mw.int().tolist())) for mw in pred_mw_from_text]
        pred_mw_sents_from_motion = [" ".join(map(str, mw.int().tolist())) for mw in pred_mw_from_motion]
        target_mw_sents = [[" ".join(map(str, mw.int().tolist()))] for mw in remove_padding(target, batch["length"])]

        self.metrics[split]['bleu_text2mw'].update(pred_mw_sents_from_text, target_mw_sents)
        self.metrics[split]['bleu_mw2mw'].update(pred_mw_sents_from_motion, target_mw_sents)

        # Compute $$\sum PP}L(X)$$ where X = single sequence.
        # Since target = 1-hot, we use CE(.) to compute -log p_{gt}

        # TODO: Define somewhere else
        CE = nn.CrossEntropyLoss(reduction='none')
        # Shape (b_sz, T). Result is CE_t for each time-step t=[1:T]
        ce_tensor = CE(probs_from_motion, target)
        ppl = torch.exp(ce_tensor.mean(dim=-1))  # Sum CE_t for all t=[1:T]
        self.metrics[split]['ppl_mw2mw'].update(ppl.mean().cpu())

        return loss

    def allsplit_epoch_end(self, split: str, outputs):
        losses = self.losses[split]
        loss_dict = losses.compute(split)
        dico = {losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items()}

        #Accuracy and BLEU
        metrics_dict = {f"Metrics/{name}/{split}": metric.compute() for name, metric in self.metrics[split].items()}
        # Perplexity (geometric mean over batch)
        metrics_dict[f"Metrics/ppl_text2mw/{split}"] = torch.exp(loss_dict['recons_text2mw'])
        metrics_dict[f"Metrics/ppl_mw2mw/{split}"] = torch.exp(loss_dict['recons_mw2mw'])
        dico.update(metrics_dict)

        # if split == "val":
        #     # pdb.set_trace()
        #     # metrics_dict = self.metrics.compute()
        #     # metrics_dict = {key:metrics_dict[key] for key in metrics_dict.keys() if key not in ['APE_joints', 'APE_pose', 'AVE_joints', 'AVE_pose']}
        #     # dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})
        #     accuracy = self.metrics.compute()
        #     dico.update({"Metrics/Accuracy/train": accuracy})

        dico.update({"epoch": float(self.trainer.current_epoch),
                    "step": float(self.trainer.current_epoch)})
        self.log_dict(dico)
