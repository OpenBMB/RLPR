# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import io
import PIL
import re
import os
import uuid
import math
import re
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from collections import Counter
from typing import Type, Dict, List
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import random
from tqdm import tqdm
from omegaconf import ListConfig
import copy
import json

import torch.nn.functional as F

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score.repetition import detect_repetition_with_hash



WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]] # global_pool_id -> available gpus
    mapping: dict[Role, str]   # Role -> global_pool_id
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, pr_weight=None, vr_weight=1.0):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns, filter_rate, stds = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        data.batch['final_reward_stds'] = stds
    elif adv_estimator == 'prnostd_vrstd':
        assert pr_weight is not None, "prnostd_vrstd is only for soft mix reward"
        token_level_pr = data.batch['token_level_pr']
        token_level_vr = data.batch['token_level_vr']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns, filter_rate, stds = core_algos.compute_prnostd_vrstd_outcome_advantage(token_level_pr=token_level_pr,
                                                                        token_level_vr=token_level_vr,
                                                                        pr_weight=pr_weight,
                                                                        vr_weight=vr_weight,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        data.batch['final_reward_stds'] = stds
    elif adv_estimator == 'prstd_vrstd':
        assert pr_weight is not None, "prnostd_vrstd is only for soft mix reward"
        token_level_pr = data.batch['token_level_pr']
        token_level_vr = data.batch['token_level_vr']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns, filter_rate, pr_stds, vr_stds = core_algos.compute_prstd_vrstd_outcome_advantage(token_level_pr=token_level_pr,
                                                                        token_level_vr=token_level_vr,
                                                                        pr_weight=pr_weight,
                                                                        vr_weight=vr_weight,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        data.batch['final_reward_stds'] = pr_stds
    elif adv_estimator == 'grpo_nostd':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns, filter_rate, stds = core_algos.compute_grpo_nostd_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        data.batch['final_reward_stds'] = stds
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data, filter_rate


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._decode_cache = {}
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.chat_template = self.tokenizer.chat_template
        self.decode_hit = 0
        self.decode_miss = 0

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return self.tokenizer(text, **kwargs)
        return_tensors = kwargs.pop("return_tensors", None)
        raw_inputs = self.tokenizer(text, **kwargs, padding=True)
        inputs = {key: value if key != "input_ids" and key != "attention_mask" else [] for key, value in
                  raw_inputs.items()}
        for i in range(len(text)):
            if return_tensors == "pt":
                inputs["input_ids"].append(torch.tensor(raw_inputs["input_ids"][i]))
                inputs["attention_mask"].append(torch.tensor(raw_inputs["attention_mask"][i]))
            else:
                inputs["input_ids"].append(raw_inputs["input_ids"][i])
                inputs["attention_mask"].append(raw_inputs["attention_mask"][i])
        if return_tensors == "pt":
            inputs["input_ids"] = torch.tensor(raw_inputs["input_ids"])
            inputs["attention_mask"] = torch.tensor(raw_inputs["attention_mask"])
        return inputs

    def decode(self, input_ids, **kwargs):
        # Convert input_ids to a tuple for hashing if it's a tensor
        if hasattr(input_ids, "tolist"):
            cache_key = tuple(input_ids.tolist())
        else:
            cache_key = tuple(input_ids)

        # Add kwargs to cache key to ensure different kwargs get different cache entries
        cache_key = (cache_key, tuple(sorted(kwargs.items())))

        if cache_key in self._decode_cache:
            self.decode_hit += 1
            return self._decode_cache[cache_key]

        self.decode_miss += 1
        result = self.tokenizer.decode(input_ids, **kwargs)
        self._decode_cache[cache_key] = result
        return result

    def batch_decode(self, input_ids, **kwargs):
        return [
            self.decode(input_ids[i], **kwargs) for i in range(len(input_ids))
        ]

    def release_cache(self):
        self._decode_cache = {}
        self.decode_hit = 0
        self.decode_miss = 0

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, **kwargs)

class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ['grpo', 'grpo_nostd', 'reinforce_plue_plus', 'remax', 'rloo', 'prnostd_vrstd', 'prstd_vrstd']:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataset_for_llama_and_gemma()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataset_for_llama_and_gemma(self):
        # We create datasets for Llama and Gemma
        if any(text in self.config.actor_rollout_ref.model.path.lower() for text in ['llama', 'gemma']):
            # llama/gemma tokenizer does not support truncation, so we use 'error' to raise error when the prompt is too long
            # for files in [eval(self.config.data.train_files), eval(self.config.data.val_files)]:
            
            assert self.config.reward_model.get('format_mode', 'R1') == 'R1_nothink' # we only support R1_nothink for llama and gemma
            self.config.data.train_files = self.config.data.train_files if isinstance(self.config.data.train_files, (List, ListConfig)) else [self.config.data.train_files]
            self.config.data.val_files = self.config.data.val_files if isinstance(self.config.data.val_files, (List, ListConfig)) else [self.config.data.val_files]
            for files in [self.config.data.train_files, self.config.data.val_files]:
                for fn in files:
                    df = pd.read_parquet(fn)
                    def modify_prompt(row):
                        row = row[1:] # remove system prompt
                        row[0]['content'] = row[0]['content'] + "\nPlease reason step by step, and put your final answer within <answer> </answer>."
                        if any(text in fn for text in ['MMLUPro', 'gpqa_diamond', 'WebInstruct-verified', 'SuperGPQA']):
                            row[0]['content'] = row[0]['content'] + '\nPlease only provide the letter of the answer in the tags.'
                        return row
                    df['prompt'] = df['prompt'].apply(modify_prompt)
                    save_path = fn.replace('/test/', '/test_llama_and_gemma/') if 'test' in fn else fn.replace('/train/', '/train_llama_and_gemma/')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    assert save_path != fn, f"{save_path=} {fn=}"
                    print(f"Saving modified prompts to {save_path}")
                    df.to_parquet(save_path, index=False)
            self.config.data.train_files = [file.replace('/train/', '/train_llama_and_gemma/') for file in self.config.data.train_files if file is not None]
            self.config.data.val_files = [file.replace('/test/', '/test_llama_and_gemma/') for file in self.config.data.val_files if file is not None]


    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size


        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            print(f"We shuffle the training data...")
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            print(f"We do not shuffle the training data...")
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        if 'prob' in self.config.reward_model.reward_manager:
            from torch.utils.data import ConcatDataset
            train_dataset_repeat = ConcatDataset([
                RLHFDataset(parquet_files=self.config.data.train_files,
                                            tokenizer=self.tokenizer,
                                            prompt_key=self.config.data.prompt_key,
                                            max_prompt_length=self.config.data.max_prompt_length,
                                            filter_prompts=True,
                                            return_raw_chat=self.config.data.get('return_raw_chat', False),
                                            truncation='error')
                for _ in range(2)
            ])
            sampler_repeat = SequentialSampler(data_source=train_dataset_repeat)
            self.train_dataloader_repeat = DataLoader(dataset=train_dataset_repeat,
                                            batch_size=self.config.data.train_batch_size,
                                            drop_last=False,
                                            collate_fn=collate_fn,
                                            sampler=sampler_repeat)

        print(f"{self.config.data.val_files=} {type(self.config.data.val_files)=}")
        if not isinstance(self.config.data.val_files, (List, ListConfig)):
            parquet_files = [self.config.data.val_files]
        else:
            parquet_files = self.config.data.val_files

        parquet_files = copy.deepcopy(parquet_files)
        # self.config.data.val_files = self.config.data.val_files if isinstance(self.config.data.val_files, list) else [self.config.data.val_files]
        self.val_datasets = []
        self.val_dataloaders = []
        self.val_names = []
        for idx, val_file in enumerate(parquet_files):
            print(f"Working on {val_file=}")
            # Create dataset for current file
            val_dataset = RLHFDataset(
                parquet_files=val_file,  
                tokenizer=self.tokenizer,
                prompt_key=self.config.data.prompt_key,
                max_prompt_length=self.config.data.get('val_max_prompt_length', 3072),
                filter_prompts=False,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation='error'
            )
            
            # Create dataloader for current dataset
            val_dataloader = DataLoader(
                dataset=val_dataset,
                batch_size=len(val_dataset),  # Use full dataset size as batch size
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn
            )
            
            self.val_datasets.append(val_dataset)
            self.val_dataloaders.append(val_dataloader)
            self.val_names.append(os.path.basename(val_file).replace('.', '_'))



        assert len(self.train_dataloader) >= 1
        for i in range(len(self.val_dataloaders)):
            assert len(self.val_dataloaders[i]) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Total number of training samples: {len(self.train_dataloader.dataset)}')
        for i, val_dataloader in enumerate(self.val_dataloaders):
            print(f'Size of val dataloader {i+1}/{len(self.val_dataloaders)}: {len(val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores, accs, format_rewards, data_sources, 
                                            extracted_answers, ground_truths,
                                            table_attr_name, table_name):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        grouped_samples = defaultdict(list)
        for sample in zip(data_sources, inputs, outputs, extracted_answers, ground_truths, scores, accs, format_rewards):
            grouped_samples[sample[0]].append(sample)

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)

        # Randomly select `generations_to_log` samples from each data_source
        selected_samples = []
        for data_source, samples in grouped_samples.items():
            rng.shuffle(samples)
            selected_samples.extend(samples[:generations_to_log])
        samples = selected_samples


        # Create column names for all samples
        columns = ["step"] + sum([[f"{i+1}_data_source", f"{i+1}_inputs", f"{i+1}_outputs", f"{i+1}_extracted_answer", f"{i+1}_ground_truth", f"{i+1}_score", f"{i+1}_acc", f"{i+1}_format_reward"] for i in range(len(samples))], [])


        if not hasattr(self, table_attr_name):
            # Initialize the table on first call
            setattr(self, table_attr_name, wandb.Table(columns=columns))

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=getattr(self, table_attr_name).data)


        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({f"val/{table_name}": new_table}, step=self.global_steps)
        setattr(self, table_attr_name, new_table)

    def _maybe_log_filtered_samples_to_wandb(self, table_attr_name, table_name, filtered_out_batch: DataProto, N=3):
        """Log a table of filtered samples to wandb"""
        """filtered_out_batch: {
            'scoreAs': scoreA_tensor[~final_mask],
            'scoreBs': scoreB_tensor[~final_mask],
            'rewards': reward_tensor[~final_mask],
            'format_rewards': format_reward_tensor[~final_mask],
            'extracted_answers': [ans for index, ans in enumerate(extracted_answer_list) if final_mask[index] == False],
            'pr_scoreAs': pr_scoreA_tensor[~final_mask],
            'pr_scoreBs': pr_scoreB_tensor[~final_mask],
            'accs': exact_tensor[~final_mask],
        }"""
        generations_to_log = self.config.trainer.get("filtered_samples_to_log_to_wandb", 0)

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `filtered_samples_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        n_samples = self.config.actor_rollout_ref.rollout.n
        N = min(N * n_samples, len(filtered_out_batch))
        assert len(filtered_out_batch) % n_samples == 0

        batch = filtered_out_batch
        data_sources = [batch[i_].non_tensor_batch['data_source'] for i_ in range(N)]
        extracted_answers = [batch[i_].non_tensor_batch['extracted_answers'] for i_ in range(N)]
        scoreAs = [batch[i_].batch['scoreAs'].sum(-1).item() for i_ in range(N)]
        scoreBs = [batch[i_].batch['scoreBs'].sum(-1).item() for i_ in range(N)]
        rewards = [batch[i_].batch['rewards'].sum(-1).item() for i_ in range(N)]
        format_rewards = [batch[i_].batch['format_rewards'].sum(-1).item() for i_ in range(N)]
        pr_scoreAs = [batch[i_].batch['pr_scoreAs'].sum(-1).item() for i_ in range(N)] if 'pr_scoreAs' in batch[0].batch else [0] * N
        pr_scoreBs = [batch[i_].batch['pr_scoreBs'].sum(-1).item() for i_ in range(N)] if 'pr_scoreBs' in batch[0].batch else [0] * N
        accs = [batch[i_].batch['accs'].sum(-1).item() for i_ in range(N)] if 'accs' in batch[0].batch else [0] * N
        sequences = [self.tokenizer.decode(batch[i_].batch['input_ids'][batch[i_].batch['attention_mask'].bool()], skip_special_tokens=False) for i_ in range(N)]
        ground_truths = [batch[i_].non_tensor_batch['reward_model']['ground_truth'] for i_ in range(N)]


        if 'mix' in self.config.reward_model.reward_manager:
            samples = list(zip(
                data_sources, extracted_answers, ground_truths, rewards, scoreBs, scoreAs, accs, pr_scoreBs, pr_scoreAs, format_rewards, sequences
            ))
        else:
            samples = list(zip(
                data_sources, extracted_answers, ground_truths, rewards, scoreBs, scoreAs, format_rewards, sequences
            ))

        columns = ['step']
        for i in range(N):
            j = i // n_samples + 1
            k = i % n_samples + 1
            columns += [f'{j}_{k}_data_source']
            columns += [f'{j}_{k}_extracted_answer']
            columns += [f'{j}_{k}_ground_truth']
            columns += [f'{j}_{k}_reward']
            columns += [f'{j}_{k}_scoreB']
            columns += [f'{j}_{k}_scoreA']
            if 'mix' in self.config.reward_model.reward_manager:
                columns += [f'{j}_{k}_acc']
                columns += [f'{j}_{k}_pr_scoreB']
                columns += [f'{j}_{k}_pr_scoreA']
            columns += [f'{j}_{k}_format_reward']
            columns += [f'{j}_{k}_sequence']

        if not hasattr(self, table_attr_name):
            # Initialize the table on first call
            setattr(self, table_attr_name, wandb.Table(columns=columns))

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        # new_table = wandb.Table(columns=columns, data=self.train_table.data)
        new_table = wandb.Table(columns=columns, data=getattr(self, table_attr_name).data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({f"filter/{table_name}": new_table}, step=self.global_steps)
        setattr(self, table_attr_name, new_table)


    def _maybe_log_train_generations_to_wandb(self, sequences, advantages, scores, scoreAs, scoreBs, 
                                              format_rewards, final_reward_stds, entropies, data_sources, extracted_answers, ground_truths,
                                              vr_scores=None, pr_scoreAs=None, pr_scoreBs=None,
                                              table_attr_name=None, table_name=None):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.get("train_generations_to_log_to_wandb", 0)

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `train_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        if 'mix' in self.config.reward_model.reward_manager:
            samples = list(zip(data_sources, sequences, extracted_answers, ground_truths, advantages, scores, scoreAs, scoreBs, format_rewards, final_reward_stds, entropies, vr_scores, pr_scoreAs, pr_scoreBs))
            columns = ["step"] + sum([[f"{i+1}_data_source", f"{i+1}_sequence", f"{i+1}_extracted_answer", f"{i+1}_ground_truth", f"{i+1}_advantage", f"{i+1}_score", f"{i+1}_scoreA", f"{i+1}_scoreB", f"{i+1}_format_reward", f"{i+1}_final_reward_std", f"{i+1}_entropy", f"{i+1}_vr_score", f"{i+1}_pr_socreA", f"{i+1}_pr_scoreB"] for i in range(len(samples))], [])
        else:
            samples = list(zip(data_sources, sequences, extracted_answers, ground_truths, advantages, scores, scoreAs, scoreBs, format_rewards, final_reward_stds, entropies))
            columns = ["step"] + sum([[f"{i+1}_data_source", f"{i+1}_sequence", f"{i+1}_extracted_answer", f"{i+1}_ground_truth", f"{i+1}_advantage", f"{i+1}_score", f"{i+1}_scoreA", f"{i+1}_scoreB", f"{i+1}_format_reward", f"{i+1}_final_reward_std", f"{i+1}_entropy"] for i in range(len(samples))], [])
        samples.sort(key=lambda x: x[0])  # Sort by input text


        if not hasattr(self, table_attr_name):
            # Initialize the table on first call
            setattr(self, table_attr_name, wandb.Table(columns=columns))

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=getattr(self, table_attr_name).data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({f"train/{table_name}": new_table}, step=self.global_steps)
        setattr(self, table_attr_name, new_table)

    def _maybe_log_histogram_to_wandb(self, values, histogram_name, column_name, title, bins=20):
        import wandb
        if 'wandb' not in self.config.trainer.logger:
            return

        if len(values) > 10000:
            values = random.sample(values, 10000)

        plt.figure(figsize=(8, 6))
        weights = np.ones_like(values) / len(values) * 100  # Calculate percentages
        plt.hist(values, bins=bins, edgecolor='black', alpha=0.7, weights=weights)
        plt.title(title)
        plt.xlabel("Values")
        plt.ylabel(f"Percentage (%) of {column_name}")
        plt.grid(axis='y', alpha=0.5)
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f%%'))
        img_buf = io.BytesIO()
        plt.savefig(img_buf)
        image = PIL.Image.open(img_buf)
        wandb.log({histogram_name + f"_step{self.global_steps}": [wandb.Image(image)]}, step=self.global_steps)
        plt.close()

        
        # Save histogram to data/logs/val_results
        save_dir = "data/logs/val_results"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, histogram_name.split('/')[0]), exist_ok=True)
        
        # Create and save the histogram using matplotlib
        plt.figure()
        plt.hist(values, bins=20)
        plt.title(title + f'_step{self.global_steps}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        
        # Save the plot
        file_name = f"{histogram_name}_step{self.global_steps}.png"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path)
        plt.close()

    def _validate(self):
        result = {}
        test_decoding_strategy = self.config.trainer.get('test_decoding_strategy', 'sampling+greedy')

        if 'sampling' in test_decoding_strategy:
            result_sampling = self._validate_inner(decoding_strategy='sampling')
            result.update(result_sampling)
        if 'greedy' in test_decoding_strategy:
            result_greedy = self._validate_inner(decoding_strategy='greedy')
            result.update(result_greedy)
        if 'sampling' in test_decoding_strategy and 'greedy' in test_decoding_strategy:
            for data_source_set in [
                ['MMLUPro-1000_Avg2', 'gpqa_diamond_Avg4',  'TheoremQA_Avg2','WebInstruct-verified-val_Avg2'],
            ]:
                result_best = []
                for val_set in data_source_set:
                    pattern = r'val_(greedy|sampling)/test_acc/' + val_set + r'-.*' 
                    acc_list = []
                    for k, v in result.items():
                        if re.match(pattern, k):
                            acc_list.append(v)
                    if len(acc_list) != 0:
                        result_best.append(max(acc_list))
                if len(result_best) == len(data_source_set):
                    result.update({f'val_mean/test_acc/best-mean_{"+".join([source for source in data_source_set])}': sum(result_best) / len(result_best)})
                print(f"{result_best=} {data_source_set=}")
        return result



    def _validate_inner(self, decoding_strategy):
        acc_tensor_lst = []
        data_source_lst = []
        data_source2count = defaultdict(int)
        data_source_set = set()

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_accs = []
        sample_format_rewards = []
        sample_extracted_answers = []
        sample_ground_truths = []
        sample_data_sources = []

        if not hasattr(self, 'val_dataloaders'):
            self.val_dataloaders = [self.val_dataloader]

        with tqdm(total=len(self.val_dataloaders)) as pbar:
            for idx, val_dataloader in enumerate(self.val_dataloaders):
                val_name = self.val_names[idx]
                match = re.search(r'Avg(\d*)', val_name)
                if match and decoding_strategy == 'sampling':
                    n = int(match.group(1))
                    print(f"We rollout {n=} samples for {val_name}")
                else:
                    n = 1
                print(f'\n## validating {val_name} ##\n')
                for test_data in val_dataloader:
                    test_batch = DataProto.from_single_dict(test_data)

                    # we only do validation on rule-based rm
                    if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                        return {}

                    test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                    if decoding_strategy == 'greedy':
                        test_gen_batch.meta_info = {
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'pad_token_id': self.tokenizer.pad_token_id,
                            'recompute_log_prob': False,
                            'do_sample': False,
                            'validate': True,
                            'n': 1,
                        }
                    elif decoding_strategy == 'sampling':
                        test_gen_batch.meta_info = {
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'pad_token_id': self.tokenizer.pad_token_id,
                            'recompute_log_prob': False,
                            'do_sample': True,
                            'validate': True,
                            # 'n': 1,
                            'n': n,
                            'seed': self.config.trainer.get('val_seed', 42),
                            'top_p': self.config.actor_rollout_ref.rollout.get('val_top_p', self.config.actor_rollout_ref.rollout.top_p),
                        }
                    else:
                        raise ValueError

                    # Store original inputs
                    input_ids = test_gen_batch.batch['input_ids']
                    input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                    input_texts = [item for item in input_texts for _ in range(test_gen_batch.meta_info['n'])]
                    sample_inputs.extend(input_texts)



                    print(f"In validation: {test_gen_batch.meta_info=}", flush=True)

                    # pad to be divisible by dp_size
                    test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)

                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                    test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size * n)


                    # Store generated outputs
                    output_ids = test_output_gen_batch.batch['responses']
                    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    sample_outputs.extend(output_texts)
                    
                    test_batch = test_batch.repeat(repeat_times=n, interleave=True)
                    test_batch = test_batch.union(test_output_gen_batch)

                    # evaluate using reward_function
                    reward_tensor, acc_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list = self.val_reward_fn(test_batch, name=f"{self.config.trainer.experiment_name}-{val_name}-{decoding_strategy}-global_step_{self.global_steps}")

                    # Store scores
                    scores = reward_tensor.sum(-1).cpu().tolist()
                    accs = acc_tensor.sum(-1).cpu().tolist()
                    format_rewards = format_reward_tensor.sum(-1).cpu().tolist()
                    sample_scores.extend(scores)
                    sample_accs.extend(accs)
                    sample_format_rewards.extend(format_rewards)
                    sample_extracted_answers.extend(extracted_answer_list)
                    for i_ in range(len(test_batch)):
                        sample_ground_truths.append(test_batch[i_].non_tensor_batch['reward_model']['ground_truth'])
                        sample_data_sources.append(test_batch[i_].non_tensor_batch['data_source'])

                    acc_tensor_lst.append(acc_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * acc_tensor.shape[0]))
                    for i in range(len(test_batch)):
                        data_source2count[test_batch[i].non_tensor_batch.get('data_source', 'unknown')] += 1
                        data_source_set.add(test_batch[i].non_tensor_batch.get('data_source', 'unknown'))
                pbar.update(1)
        print(f"{data_source2count=}")

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, 
                                                 scores=sample_scores, accs=sample_accs, 
                                                 format_rewards=sample_format_rewards, data_sources=sample_data_sources, 
                                                 extracted_answers=sample_extracted_answers, 
                                                 ground_truths=sample_ground_truths,
                                                 table_attr_name=f'val_table_{decoding_strategy}',
                                                 table_name=f"val_generations_{decoding_strategy}")

 
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        data_source_acc = {}
        for i in range(acc_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_acc:
                data_source_acc[data_source] = []
            data_source_acc[data_source].append(acc_tensor[i].item())



        metric_dict = {}
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val_{decoding_strategy}/test_acc/{data_source}'] = np.mean(accs)

        # Compute overall mean score across all data sources
        suffixes = ['R1', 'OC', 'SysR1', 'UserR1']  # Add or remove suffixes as needed
        

        for data_source_set in [
            ['MMLUPro-1000_Avg2', 'gpqa_diamond_Avg4',  'TheoremQA_Avg2','WebInstruct-verified-val_Avg2'],
        ]:
            for suffix in suffixes:
                self.compute_and_store_means(metric_dict, data_source_set, suffix, decoding_strategy)

            # Compute the best accuracy for each data source and store the mean
            acc_list = []
            for val_set in data_source_set:
                for suffixes_log in [['OC', 'SysR1'], ['OC', 'UserR1'], ['SysR1']]:
                    if all(f'val_{decoding_strategy}/test_acc/{val_set}-{suffix}' in metric_dict for suffix in suffixes_log):
                        acc_list.append(max(metric_dict[f'val_{decoding_strategy}/test_acc/{val_set}-{suffix}'] for suffix in suffixes_log))
                    if len(acc_list) > 0:
                        metric_dict[f'val_mean_{decoding_strategy}/test_acc/best-mean_{"+".join([source for source in data_source_set])}'] = np.mean(acc_list)

        return metric_dict

    def compute_and_store_means(self, metric_dict, data_source_set, suffix, decoding_strategy=None):
        specific_mean = self.compute_specific_means(metric_dict, [f'{source}-{suffix}' for source in data_source_set], decoding_strategy=decoding_strategy)
        if specific_mean is not None:
            if decoding_strategy:
                metric_dict[f'val_mean_{decoding_strategy}/test_acc/{suffix}-mean_{"+".join([source for source in data_source_set])}'] = specific_mean
            else:
                metric_dict[f'val_mean/test_acc/{suffix}-mean_{"+".join([source for source in data_source_set])}'] = specific_mean

    @staticmethod
    def compute_specific_means(metric_dict, specific_sources, decoding_strategy=None):
        """
        Compute the mean score across specific data sources.

        Args:
            metric_dict (dict): A dictionary containing mean scores for each data source.
            specific_sources (list): A list of specific data source keys to compute the mean for.

        Returns:
            float: The mean score across the specified data sources, or None if none are found.
        """
        # Filter the specific means from the metric_dict
        if decoding_strategy:
            specific_means = [metric_dict[f'val_{decoding_strategy}/test_acc/{source}'] for source in specific_sources if f'val_{decoding_strategy}/test_acc/{source}' in metric_dict]
        else:
            specific_means = [metric_dict[f'val/test_acc/{source}'] for source in specific_sources if f'val/test_acc/{source}' in metric_dict]
        
        # Compute the mean if specific means are found
        if len(specific_means) == len(specific_sources):
            return np.mean(specific_means)
        else:
            return None

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

        if hasattr(self, 'ema_mean'):
            ema_mean_json_path = os.path.join(self.config.trainer.default_local_dir, 'ema_mean.json')
            save_to_json(self.ema_mean_json, ema_mean_json_path)


    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable' or self.config.trainer.get('val_only', False) is True:
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        elif self.config.trainer.resume_mode == 'resume_path':
            print(f"Resume path")
            assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
            assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)


        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}') 



        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path, weights_only=False)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

        ema_mean_json_path = os.path.join(self.config.trainer.default_local_dir, 'ema_mean.json')
        if os.path.exists(ema_mean_json_path):
            self.ema_mean_json = load_from_json(ema_mean_json_path)
            self.ema_mean = self.ema_mean_json[str(self.global_steps)]



    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        if metrics:
            global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                        partitions=global_partition_lst,
                                                        prefix=logging_prefix)
            metrics.update(global_balance_stats)

    def filter(self, reward_tensor, batch: DataProto, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.
        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt
        Returns:
            DataProto: Filtered batch
        """
        # First do accuracy filtering if enabled
        if self.config.data.get("filter_accuracy", False):
            if isinstance(reward_tensor, list):
                print("reward_tensor: ", len(reward_tensor), reward_tensor)
                reward_tensor = torch.tensor(reward_tensor)
                reward_matrix = reward_tensor.reshape(-1, n_samples)
            else:
                print("reward_tensor: ", reward_tensor.shape, reward_tensor)
                reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            print("reward_matrix: ", reward_matrix.shape, reward_matrix)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            print("acc_tensor:", acc_tensor.shape, acc_tensor)
            counts = Counter(acc_tensor.tolist())
            print("Accuracy distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())))

            acc_mask = (acc_tensor >= self.config.data.accuracy_lower_bound) & (
                        acc_tensor <= self.config.data.accuracy_upper_bound)
            numeric_valid_mask = ~(torch.isnan(acc_tensor) | torch.isinf(acc_tensor))
            acc_mask &= numeric_valid_mask
            print("acc_mask: ", acc_mask.shape, acc_mask)

            num_lt_lower_bound = (acc_tensor < self.config.data.accuracy_lower_bound).sum().item()
            num_gt_upper_bound = (acc_tensor > self.config.data.accuracy_upper_bound).sum().item()
            total_num = len(batch) // n_samples
            metrics = {
                "filter/rate": (num_lt_lower_bound + num_gt_upper_bound) / total_num,
                "filter/lower_rate": num_lt_lower_bound / total_num,
                "filter/upper_rate": num_gt_upper_bound / total_num}
            print(f"filter/rate: {(num_lt_lower_bound + num_gt_upper_bound) / total_num}, filter/lower_rate: {num_lt_lower_bound / total_num}, filter/upper_rate: {num_gt_upper_bound / total_num}")
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)
        # Then do truncation filtering if enabled
        if self.config.data.filter_truncated:
            responses = batch.batch['responses']
            attention_mask = batch.batch['attention_mask']
            response_mask = attention_mask[:, -responses.size(1):]

            response_lengths = response_mask.sum(-1)  # (batch_size,)
            response_lengths = response_lengths.reshape(-1, n_samples)  # (num_prompts, n_samples)
            max_len = self.config.data.max_response_length

            has_truncated = (response_lengths >= max_len).any(dim=-1)

            truncated_counts = Counter(has_truncated.tolist())
            print("Truncation distribution:",
                f"Truncated: {truncated_counts[True] if True in truncated_counts else 0}, "
                f"Non-truncated: {truncated_counts[False] if False in truncated_counts else 0}")
            # Keep only prompts where no response was truncated
            trunc_mask = ~has_truncated
        else:
            # If truncation filtering disabled, keep all samples
            trunc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)

        combined_mask = acc_mask & trunc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)
        filtered_batch = batch.slice(final_mask)

        print(f"Filtered batch size: {len(filtered_batch)} (from original size: {len(batch)})")
        return filtered_batch, final_mask, metrics, batch

    def get_from_cache(self, cache_data, count, dp_size):
        if len(cache_data) == count:
            return cache_data, []
        if count > len(cache_data):
            count = (len(cache_data) // dp_size) * dp_size
        samples = cache_data.slice(range(0, count))
        cache_data = cache_data.slice(range(count, len(cache_data)))
        return samples, cache_data

    def add_to_cache(self, cache_data, batch: DataProto, n_samples, tmp_dict: dict):
        # get first of n_samples
        batch = batch.slice(range(0, len(batch), n_samples))
        # notice that we only add prompts to buffer, and slicing strategy should be exactly consistent to what is in ray_trainer.py
        batch.pop(batch_keys=["responses", "prompts"])
        for k,v in batch.batch.items():
            print(f"cache {k}:", v.shape)
        batch.slice_batch(start=0, length=self.config.data.max_prompt_length, dim=-1)

        # get indices
        batch_uids = batch.non_tensor_batch.pop("uid").tolist()
        tmp_uids = tmp_dict.pop("uid").tolist()
        indices = [tmp_uids.index(uid) for uid in batch_uids]
        for k in tmp_dict.keys():
            if not k in batch.non_tensor_batch.keys():
                batch.non_tensor_batch[k] = tmp_dict[k][indices]

        if len(cache_data) == 0:
            return batch
        else:
            return DataProto.concat([cache_data, batch])


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        if not self.config.trainer.get('val_only', False):
            if 'prob' in self.config.reward_model.reward_manager:
                promptgt2scoreA = self.compute_promptgt2scoreA(0)

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        batch_size = self.config.data.train_batch_size
        n_samples = self.config.actor_rollout_ref.rollout.n

        if self.config.data.get('filter_accuracy', False) or self.config.data.get('filter_truncated', False):
            config_data_filter_accuracy, config_data_filter_truncated = self.config.data.filter_accuracy, self.config.data.filter_truncated

            if self.config.data.get('filter_mode', 'default') in ['EMA']:
                score_std_list = []

        else:
            config_data_filter_accuracy, config_data_filter_truncated = False, False

        for epoch in range(self.config.trainer.total_epochs):

            print(f"-------- epoch {epoch} --------")
            iter_dataloader = iter(self.train_dataloader)
            buffer_batch = []
            cache_data = []

            while True:
                self.global_steps += 1
                print(f"------------------------------ {self.global_steps} --------------------------------")
                if self.global_steps >= self.total_training_steps:
                    break

                metrics = {}
                timing_raw = {}

                if config_data_filter_accuracy or config_data_filter_truncated:
                    if self.config.data.get('filter_mode', 'default') in ['EMA']:
                        if self.global_steps >= self.config.data.get('filter_start_step', 0):
                            self.config.data.filter_accuracy, self.config.data.filter_truncated = config_data_filter_accuracy, config_data_filter_truncated
                        else:
                            self.config.data.filter_accuracy, self.config.data.filter_truncated = False, False
                            assert self.config.data.accuracy_lower_bound == 0


                    if self.config.data.get('filter_mode', 'default') == 'EMA':
                        if self.global_steps == 1 or (self.config.data.get('resume_ema_mean', None) is not None):
                            self.ema_mean = self.config.data.get('resume_ema_mean', None)
                            if self.ema_mean is not None:
                                print(f"We resume the ema_mean as {self.ema_mean=}")
                        if self.global_steps >= self.config.data.filter_ema_start_step:
                            if self.global_steps >= self.config.data.filter_start_step:
                                print(f"We are calculating the score_std_list using EMA: {self.global_steps=} {len(score_std_list)=} {score_std_list=}")
                                if len(score_std_list) > 0 and isinstance(score_std_list[-1], list):
                                    if len(score_std_list[-1]) == 0: # After 1 epoch, the code breaks without computing the mean. So we skip this empty list.
                                        score_std_list = score_std_list[:-1]
                                    else:
                                        score_std_list[-1] = sum(score_std_list[-1]) / len(score_std_list[-1])
                                
                                if self.config.data.get('resume_ema_mean', None):
                                    self.config.data.resume_ema_mean = None
                                else:
                                    if self.ema_mean is None:
                                        self.ema_mean = sum(score_std_list) / len(score_std_list) # init the ema
                                        self.ema_mean_json = {}
                                    elif len(score_std_list) == 0: # THe self.ema_mean is loaded from the saved txt file
                                        pass
                                    else:
                                        new_value = score_std_list[-1]
                                        if not (math.isnan(new_value) or math.isinf(new_value)):
                                            self.ema_mean = score_std_list[-1] * (1 - self.config.data.filter_ema_ratio) + self.ema_mean * self.config.data.filter_ema_ratio
                                        else:
                                            print('Encounter NaN or Inf std!', flush=True)
                                print(f"{self.ema_mean=} for {self.global_steps=}")
                                self.ema_mean_json[self.global_steps] = self.ema_mean

                                self.config.data.accuracy_lower_bound = self.ema_mean * self.config.data.std_filter_beta
                            score_std_list.append([])




                    if self.config.data.get('filter_mode', 'default') == 'default':
                        print(f"{self.global_steps=} {self.config.data.filter_accuracy=} {self.config.data.filter_truncated=} {self.config.data.accuracy_lower_bound=} {self.config.data.accuracy_upper_bound=}")
                    else:
                        print(f"{self.global_steps=} {self.config.data.filter_accuracy=} {self.config.data.filter_truncated=} {self.config.data.accuracy_lower_bound=} {self.config.data.accuracy_upper_bound=} {len(score_std_list)=}")
                    metrics.update({"filter/accuracy_lower_bound": self.config.data.accuracy_lower_bound})
                    metrics.update({"filter/accuracy_upper_bound": self.config.data.accuracy_upper_bound})

                with _timer('step', timing_raw):

                    filtered_out_batch = []
                    current_batch_idx = 0
                    # get a batch after filtering
                    while len(buffer_batch) < batch_size * n_samples:
                        # get a raw batch , generate and fill buffer_batch
                        print(f"{len(buffer_batch)=} {batch_size=} {n_samples=}")
                        if self.config.data.get('filter_cache_regenerate', False) and len(cache_data) >= batch_size:
                            batch, cache_data = self.get_from_cache(cache_data, batch_size, self.actor_rollout_wg.world_size)
                            print(f"get from cache: {len(batch)}, left {len(cache_data)}")
                        else:
                            try:
                                batch_dict = next(iter_dataloader)
                            except StopIteration:
                                break
                            batch: DataProto = DataProto.from_single_dict(batch_dict)


                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )

                        # pop those keys for generation
                        non_tensor_batch_keys = ["raw_prompt_ids"]

                        if self.config.data.get('filter_cache_regenerate', False):
                            tmp_dict = {k:batch.non_tensor_batch[k] for k in non_tensor_batch_keys+["uid"]}



                        if 'prob' in self.config.reward_model.reward_manager:
                            # Decode all input IDs in the batch at once
                            prompts = self.tokenizer.batch_decode(
                                batch.batch['input_ids'], 
                                skip_special_tokens=False
                            )
                            # prompts = [prompt.replace(self.tokenizer.pad_token, '') for prompt in prompts]
                            prompts = [replace_left_and_right_str(prompt, self.tokenizer.pad_token) for prompt in prompts]

                            # Extract ground truths for the entire batch
                            ground_truths = [item.non_tensor_batch['reward_model']['ground_truth'] for item in batch]

                            # Combine prompts and ground truths to create keys for lookup
                            prompt_gt_keys = [prompt + '\n\n\n' + gt for prompt, gt in zip(prompts, ground_truths)]

                            # Check if any prompt_gt_key is missing in promptgt2scoreA
                            if any(key not in promptgt2scoreA for key in prompt_gt_keys):
                                print("Skipping batch due to missing scoreA.")  # Log for robustness
                                continue

                            # Assign scoreA to each item in the batch
                            for i, key in enumerate(prompt_gt_keys):
                                batch[i].non_tensor_batch['reward_model']['scoreA'] = promptgt2scoreA[key]

                        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                        # pre-generate a batch
                        with _timer('gen', timing_raw):
                            print(f"{gen_batch.meta_info=}")
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        if self.config.trainer.get("obtain_rollout_only", False):
                            print(f"Get rollout for Step {self.global_steps}...")
                            ground_truth_list = [batch[i_].non_tensor_batch['reward_model']['ground_truth'] for i_ in range(len(batch))]
                            ground_truth_list = [item for item in ground_truth_list for _ in range(self.config.actor_rollout_ref.rollout.n)]
                            this_index_list = [batch[i_].non_tensor_batch['extra_info'].get('this_index', 0) for i_ in range(len(batch))]
                            this_index_list = [item for item in this_index_list for _ in range(self.config.actor_rollout_ref.rollout.n)]
                            self.save_rollout(gen_batch_output, ground_truth_list, this_index_list=this_index_list)
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(gen_batch_output)
                            buffer_batch = DataProto.concat([buffer_batch, batch]) if len(buffer_batch) > 0 else batch 
                            print(f"collected {len(buffer_batch)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} responses")
                            continue


                        metrics.update(self.compute_think_answer_length_metrics(gen_batch_output))

                        if self.config.algorithm.adv_estimator == 'remax':
                            raise NotImplementedError

                        if 'prob' in self.config.reward_model.reward_manager:
                            print(f"Using cross entropy reward...")
                            ground_truth_list = [batch[i_].non_tensor_batch['reward_model']['ground_truth'] for i_ in range(len(batch))]
                            ground_truth_list = [item for item in ground_truth_list for _ in range(self.config.actor_rollout_ref.rollout.n)]
                            gen_batch_output_pr = self.construct_new_batch_optimized(gen_batch_output, ground_truth_list)

                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)


                        if 'prob' in self.config.reward_model.reward_manager:
                            batch = batch.union(gen_batch_output_pr)

                        # do accuracy filtering and score logging
                        if self.config.data.get('filter_accuracy', False) or self.config.data.get('filter_truncated', False):
                            with _timer('verify', timing_raw):
                                if 'prob' in self.config.reward_model.reward_manager:
                                    with _timer('old_log_prob_pr', timing_raw):
                                        old_log_prob_pr = self.actor_rollout_wg.compute_log_prob_pr(batch)
                                        batch = batch.union(old_log_prob_pr) 
                                        
                                if 'mix' in self.config.reward_model.reward_manager:
                                    reward_tensor, scoreB_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list, straightA_tensor, exact_tensor, pr_scoreB_tensor, pr_scoreA_tensor, pr_reward_tensor, vr_reward_tensor = self.reward_fn(batch) # reward_tensor.shape: torch.Size([40, 1024])
                                    batch.batch['token_level_pr'] = pr_reward_tensor
                                    batch.batch['token_level_vr'] = vr_reward_tensor
                                else:
                                    reward_tensor, scoreB_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list = self.reward_fn(batch)

                                scoreA_list, scoreB_list, scoreB_minus_scoreA_list = [], [], []
                                for i_ in range(len(batch)):
                                    scoreA_ = scoreA_tensor[i_].sum().item()
                                    scoreB_ = scoreB_tensor[i_].sum().item()
                                    scoreA_list.append(scoreA_)
                                    scoreB_list.append(scoreB_)
                                    scoreB_minus_scoreA_list.append(scoreB_ - scoreA_)
                                if self.config.data.get('filter_target', 'scoreB-scoreA') == 'scoreB-scoreA':
                                    print(f"We use scoreB-scoreA for filtering")
                                    filter_target = scoreB_minus_scoreA_list
                                elif self.config.data.get('filter_target', 'scoreB-scoreA') == 'scoreB':
                                    print(f"We use scoreB for filtering")
                                    filter_target = scoreB_list
                                elif self.config.data.get('filter_target', 'scoreB-scoreA') == 'final_reward_std':
                                    print(f"We use std for filtering")
                                    if self.config.reward_model.get("repetition_penalty", False):
                                        # Decode all responses in a batch
                                        responses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
                                        for i_, response_i in enumerate(responses):
                                            # Apply repetition penalty
                                            non_zero_indices = reward_tensor[i_].nonzero(as_tuple=True)
                                            repetition_penalty = detect_repetition_with_hash(response_i, window_size=10, max_repetitions_limit=self.config.reward_model.get("repetition_penalty_max_repetitions_limit", 10))
                                            reward_tensor[i_][non_zero_indices] += repetition_penalty

                                    batch.batch['token_level_scores'] = reward_tensor
                                    metrics.update(self.compute_std_per_group_merics(reward_tensor, batch, prefix=f"critic_during_filter/rewards_{current_batch_idx}"))

                                    # compute rewards. apply_kl_penalty if available
                                    if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                                        batch, kl_metrics = apply_kl_penalty(batch,
                                                                            kl_ctrl=self.kl_ctrl,
                                                                            kl_penalty=self.config.algorithm.kl_penalty)
                                    else:
                                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                                    # compute advantages, executed on the driver process
                                    batch, filter_rate = compute_advantage(batch,
                                                            adv_estimator=self.config.algorithm.adv_estimator,
                                                            gamma=self.config.algorithm.gamma,
                                                            lam=self.config.algorithm.lam,
                                                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                            pr_weight=self.config.reward_model.get('pr_weight', None),
                                                            vr_weight=self.config.reward_model.get('vr_weight', 1.0))
                                    # filter_target = final_reward_std_list
                                    # final_reward_std_ = batch[i_].batch['final_reward_stds'].item()
                                    filter_target = [batch[i_].batch['final_reward_stds'].item() for i_ in range(len(batch))]

                                    if self.config.data.get('std_filter_beta_per_data_source', None):
                                        std_filter_beta_per_data_source = self.config.data.get('std_filter_beta_per_data_source', None) # WebInstruct-verified_is_1+DAPO-Math-17k_is_1.25
                                        # Split the string into key-value pairs
                                        kvs = [pair.split('_is_') for pair in std_filter_beta_per_data_source.split('+')]
                                        dict_std_filter_beta_per_data_source = {key: float(value) for key, value in kvs}
                                        print(f"{dict_std_filter_beta_per_data_source=}")

                                        # [data_source_ = batch[i_].non_tensor_batch['data_source']]
                                        print(f"Before scaling: {filter_target=}")
                                        print(f"Data source: {[batch[i_].non_tensor_batch['data_source'] for i_ in range(len(batch))]}")
                                        filter_target = [filter_target[i_] * dict_std_filter_beta_per_data_source[batch[i_].non_tensor_batch['data_source']]  for i_ in range(len(filter_target))]
                                        print(f"After scaling: {filter_target=}")
                                        # filter_target = [filter_target[i_] * dict_std_filter_beta_per_data_source[batch[i_].non_tensor_batch['data_source']]  for i_ in range(len(filter_target))]


                                    if self.config.data.get('filter_mode', 'default') == 'EMA':
                                        if self.global_steps >= self.config.data.filter_start_step:
                                            score_std_list[-1].extend([batch[i_].batch['final_reward_stds'].item() for i_ in range(len(batch)) if (not math.isnan(batch[i_].batch['final_reward_stds'].item()) and not  math.isinf(batch[i_].batch['final_reward_stds'].item()))])
                                        print(f"{self.global_steps=} {len(score_std_list)=} {len(score_std_list[-1])}")

                                    if 'mix' in self.config.reward_model.reward_manager:
                                        batch.pop(batch_keys=['token_level_scores', 'token_level_rewards', 'token_level_vr', 'token_level_pr', 'advantages', 'returns', 'final_reward_stds'])
                                    else:
                                        batch.pop(batch_keys=['token_level_scores', 'token_level_rewards', 'advantages', 'returns', 'final_reward_stds'])
                                elif self.config.data.filter_target == 'scoreB':
                                    print(f"We use scoreB for filtering")
                                    filter_target = [scoreB_tensor[i_].sum(-1).item() for i_ in range(len(batch))]
                                elif self.config.data.filter_target == 'vr_acc':
                                    print(f"We use vr_acc for filtering")
                                    filter_target = [exact_tensor[i_].sum(-1).item() for i_ in range(len(batch))]

                                else:
                                    raise NotImplementedError

                                scoreB = scoreB_tensor.sum(-1)
                                metrics.update({# reward
                                    f'critic_during_filter/scoreB_{current_batch_idx}/mean':
                                        torch.mean(scoreB).detach().item(),
                                    f'critic_during_filter/scoreB_{current_batch_idx}/max':
                                        torch.max(scoreB).detach().item(),
                                    f'critic_during_filter/scoreB_{current_batch_idx}/min':
                                        torch.min(scoreB).detach().item(),
                                })
                                current_batch_idx += 1
                                # filter by accuracy
                                batch, final_mask, metrics_filter, origin_batch = self.filter(filter_target, batch, n_samples)
                                metrics.update(metrics_filter)

                                filtered_out_ = origin_batch.slice(~final_mask)
                                filtered_out_.union(DataProto.from_single_dict({
                                    'scoreAs': scoreA_tensor[~final_mask],
                                    'scoreBs': scoreB_tensor[~final_mask],
                                    'rewards': reward_tensor[~final_mask],
                                    'format_rewards': format_reward_tensor[~final_mask],
                                    'extracted_answers': np.array([ans for index, ans in enumerate(extracted_answer_list) if final_mask[index] == False]),
                                }))
                                if 'mix' in self.config.reward_model.reward_manager:
                                    filtered_out_.union(DataProto.from_single_dict({
                                        'pr_scoreAs': pr_scoreA_tensor[~final_mask],
                                        'pr_scoreBs': pr_scoreB_tensor[~final_mask],
                                        'accs': exact_tensor[~final_mask],
                                    }))
                                filtered_out_batch = DataProto.concat([filtered_out_batch, filtered_out_]) if len(filtered_out_batch) > 0 else filtered_out_

                        buffer_batch = DataProto.concat([buffer_batch, batch]) if len(buffer_batch) > 0 else batch 
                        print(f"collected {len(buffer_batch)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} responses")
                    # filter done

                    if len(filtered_out_batch) > 0 and (self.global_steps - 1) % 5 == 0:
                        self._maybe_log_filtered_samples_to_wandb('filtered_out_table', 'filtered_out_samples', filtered_out_batch)
                        del filtered_out_batch

                    if len(buffer_batch) < batch_size * n_samples:
                        break # dataloader is empty
                    elif len(buffer_batch) > batch_size * n_samples:
                        count = batch_size * n_samples
                        print("left data: ", (len(buffer_batch)-count)//n_samples)
                        batch = buffer_batch.slice(range(0, count))
                        buffer_batch = buffer_batch.slice(range(count, len(buffer_batch)))
                        if self.config.data.get("filter_cache_regenerate", False):
                            if 'prob' in self.config.reward_model.reward_manager:
                                buffer_batch.pop([batch_k for batch_k in buffer_batch.batch.keys() if batch_k.endswith('_pr')])
                            cache_data = self.add_to_cache(cache_data, buffer_batch, n_samples, tmp_dict)
                            buffer_batch = []
                    else:
                        batch = buffer_batch
                        buffer_batch = []
                    if self.config.trainer.get("obtain_rollout_only", False):
                        continue

                    metrics.update(self.compute_mean_of_metrics(metrics))


                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    # Actually, below is the log_prob towards the GT
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        response_entropy = old_log_prob.pop(batch_keys=['entropy']).batch['entropy']
                        row_mean = self.compute_row_mean(response_entropy, batch.batch['attention_mask']) # (N,)
                        batch = batch.union(old_log_prob)

                        metrics.update({
                            'entropy/response_entropy/mean': torch.mean(row_mean).item(),
                            'entropy/response_entropy/max': torch.max(row_mean).item(),
                            'entropy/response_entropy/min': torch.min(row_mean).item(),
                        })

                    if 'prob' in self.config.reward_model.reward_manager and self.config.data.get("filter_accuracy", False) is False:
                        # If we use accuracy filter, then skip below
                        with _timer('old_log_prob_pr', timing_raw):
                            old_log_prob_pr = self.actor_rollout_wg.compute_log_prob_pr(batch)
                            batch = batch.union(old_log_prob_pr)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    if self.config.get('use_sft_loss', False) and self.config.get('sft_type', 'multi_task') == 'bilevel':
                        with _timer('update_actor_by_sft', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch, mode='sft_only') 
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)


                        if 'mix' in self.config.reward_model.reward_manager:
                            reward_tensor, scoreB_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list, straightA_tensor, exact_tensor, pr_scoreB_tensor, pr_scoreA_tensor, pr_reward_tensor, vr_reward_tensor = self.reward_fn(batch) # reward_tensor.shape: torch.Size([40, 1024])
                            batch.batch['token_level_pr'] = pr_reward_tensor
                            batch.batch['token_level_vr'] = vr_reward_tensor
                        else:
                            reward_tensor, scoreB_tensor, scoreA_tensor, format_reward_tensor, extracted_answer_list = self.reward_fn(batch)

                        # Each row in reward_tensor contains at most 1 element.

                        scoreA_list, scoreB_list, scoreB_minus_scoreA_list = [], [], []
                        for i_ in range(len(batch)):
                            scoreA_ = scoreA_tensor[i_].sum().item()
                            scoreB_ = scoreB_tensor[i_].sum().item()
                            scoreA_list.append(scoreA_)
                            scoreB_list.append(scoreB_)
                            scoreB_minus_scoreA_list.append(scoreB_ - scoreA_)
                        if (self.global_steps - 1) % 50 == 0:
                            self._maybe_log_histogram_to_wandb(scoreA_list, f'figures/scoreA', 'scoreA', 'Score A Distribution')
                            self._maybe_log_histogram_to_wandb(scoreB_list, f'figures/scoreB', 'scoreB', 'Score B Distribution')
                            self._maybe_log_histogram_to_wandb(scoreB_minus_scoreA_list, f'figures/scoreB-scoreA', 'scoreB-scoreA', 'ScoreB - ScoreA Distribution')

                        if self.config.reward_model.get("repetition_penalty", False):
                            # Decode all responses in a batch
                            responses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
                            repetition_penalty_list = []
                            for i_, response_i in enumerate(responses):
                                # Apply repetition penalty
                                non_zero_indices = reward_tensor[i_].nonzero(as_tuple=True)
                                repetition_penalty = detect_repetition_with_hash(response_i, window_size=10, max_repetitions_limit=self.config.reward_model.get("repetition_penalty_max_repetitions_limit", 10))
                                reward_tensor[i_][non_zero_indices] += repetition_penalty
                                repetition_penalty_list.append(repetition_penalty)
                            repetition_penalty_rate = sum([1 for i in range(len(repetition_penalty_list)) if repetition_penalty_list[i] != 0]) / len(repetition_penalty_list)
                            metrics.update({"critic/repetition_penalty_rate": repetition_penalty_rate})


                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch, filter_rate = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  pr_weight=self.config.reward_model.get('pr_weight', None),
                                                  vr_weight=self.config.reward_model.get('vr_weight', 1.0))
                    metrics.update({"critic/filter_rate": filter_rate})
                    # compute reward distribution inside a batch
                    uid_to_distribution = self.compute_reward_distributions_by_group(reward_tensor, batch)
                    std_per_group, gap_per_group, mean_per_group = [], [], []
                    for group_idx, (uid, distribution) in enumerate(uid_to_distribution.items()):
                        mean_per_group.append(distribution['mean'])
                        std_per_group.append(distribution['std'])
                        gap_per_group.append(distribution['gap'])
                    
                    metrics.update({
                        "critic/rewards/mean_per_group/mean": sum(mean_per_group) / len(mean_per_group),
                        "critic/rewards/mean_per_group/max": max(mean_per_group),
                        "critic/rewards/mean_per_group/min": min(mean_per_group),
                        "critic/rewards/std_per_group/mean": sum(std_per_group) / len(std_per_group),
                        "critic/rewards/std_per_group/max": max(std_per_group),
                        "critic/rewards/std_per_group/min": min(std_per_group),
                    })
                    # uid_to_rewards = defaultdict(list)

                    ################################################
                    # Log generations to wandb & local file system
                    log_to_wandb = {
                        "scores": [],
                        "scoreAs": [],
                        "scoreBs": [],
                        "advantages": [],
                        "entropies": [],
                        "format_rewards": [],
                        "data_sources": [],
                        "sequences": [],
                        "extracted_answers": [],
                        "ground_truths": [],
                        "final_reward_stds": [],
                    }
                    if 'mix' in self.config.reward_model.reward_manager:
                        log_to_wandb.update({
                            "vr_scores": [],
                            "pr_scoreAs": [],
                            "pr_scoreBs": [],
                        })
                    entropy_list, advantage_list = [], []
                    for i_ in range(len(batch)):
                        advantage_ = torch.masked_select(batch[i_].batch['advantages'], batch[i_].batch['attention_mask'][-self.config.data.max_response_length:].bool()).mean().item()
                        advantage_list.append(advantage_)
                        entropy_ = response_entropy[i_].mean().item()
                        entropy_list.append(entropy_)
                        if batch[i_].non_tensor_batch['uid'] == batch[0].non_tensor_batch['uid']:
                            score_ = reward_tensor[i_].sum().item()
                            scoreA_ = scoreA_tensor[i_].sum().item()
                            scoreB_ = scoreB_tensor[i_].sum().item()
                            # original_scoreB_ = original_scoreB_tensor[i_].sum().item()
                            format_reward_ = format_reward_tensor[i_].sum().item()
                            data_source_ = batch[i_].non_tensor_batch['data_source']
                            sequence_ = self.tokenizer.decode(batch.batch['input_ids'][i_][batch.batch['attention_mask'][i_].bool()], skip_special_tokens=False)
                            extracted_answer_ = extracted_answer_list[i_]
                            ground_truth_ = batch[i_].non_tensor_batch['reward_model']['ground_truth']
                            final_reward_std_ = batch[i_].batch['final_reward_stds'].item()

                            log_to_wandb['scores'].append(score_)
                            log_to_wandb['scoreAs'].append(scoreA_)
                            log_to_wandb['scoreBs'].append(scoreB_)
                            log_to_wandb['advantages'].append(advantage_)
                            log_to_wandb['entropies'].append(entropy_)
                            log_to_wandb['format_rewards'].append(format_reward_)
                            log_to_wandb['data_sources'].append(data_source_)
                            log_to_wandb['sequences'].append(sequence_)
                            log_to_wandb['extracted_answers'].append(extracted_answer_)
                            log_to_wandb['ground_truths'].append(ground_truth_)
                            log_to_wandb['final_reward_stds'].append(final_reward_std_)
                            if 'mix' in self.config.reward_model.reward_manager:
                                vr_score_ = exact_tensor[i_].sum().item()
                                pr_scoreA_ = pr_scoreA_tensor[i_].sum().item()
                                pr_scoreB_ = pr_scoreB_tensor[i_].sum().item()
                                log_to_wandb['vr_scores'].append(vr_score_)
                                log_to_wandb['pr_scoreAs'].append(pr_scoreA_)
                                log_to_wandb['pr_scoreBs'].append(pr_scoreB_)

                    # if (self.global_steps - 1) % 2 == 0:
                    if (self.global_steps - 1) % 10 == 0:
                        self._maybe_log_train_generations_to_wandb(table_attr_name='train_table', table_name='generations_same_instruction', **log_to_wandb)
                    # if (self.global_steps - 1) % 10 == 0:
                    if (self.global_steps - 1) % 20 == 0:
                        N = self.config.trainer.get("train_generations_to_log_to_wandb_2", 0)
                        if N > 0:
                            if 'mix' in self.config.reward_model.reward_manager:
                                self._maybe_log_train_generations_to_wandb(table_attr_name="train_table_2", table_name="generations_varied_instruction", 
                                                                        **self.sample_batch_data(batch, reward_tensor=reward_tensor, scoreA_tensor=scoreA_tensor,
                                                                        scoreB_tensor=scoreB_tensor, advantage_list=advantage_list, entropy_list=entropy_list,
                                                                        format_reward_tensor=format_reward_tensor,
                                                                        extracted_answer_list=extracted_answer_list, 
                                                                        N=N,
                                                                        exact_tensor=exact_tensor, pr_scoreB_tensor=pr_scoreB_tensor, pr_scoreA_tensor=pr_scoreA_tensor))
                            else:
                                self._maybe_log_train_generations_to_wandb(table_attr_name="train_table_2", table_name="generations_varied_instruction", 
                                                                        **self.sample_batch_data(batch, reward_tensor=reward_tensor, scoreA_tensor=scoreA_tensor,
                                                                        scoreB_tensor=scoreB_tensor, advantage_list=advantage_list, entropy_list=entropy_list,
                                                                        format_reward_tensor=format_reward_tensor,
                                                                        extracted_answer_list=extracted_answer_list, 
                                                                        N=N))
                    # Log to /data/logs/train_generations/xxx.csv
                    if 'mix' in self.config.reward_model.reward_manager:
                        self.log_train_generations(batch, reward_tensor=reward_tensor, scoreA_tensor=scoreA_tensor,
                                                    scoreB_tensor=scoreB_tensor, advantage_list=advantage_list, entropy_list=entropy_list,
                                                    format_reward_tensor=format_reward_tensor,
                                                    extracted_answer_list=extracted_answer_list,
                                                    exact_tensor=exact_tensor, pr_scoreB_tensor=pr_scoreB_tensor, pr_scoreA_tensor=pr_scoreA_tensor)
                    else:
                        self.log_train_generations(batch, reward_tensor=reward_tensor, scoreA_tensor=scoreA_tensor,
                                                    scoreB_tensor=scoreB_tensor, advantage_list=advantage_list, entropy_list=entropy_list,
                                                    format_reward_tensor=format_reward_tensor,
                                                    extracted_answer_list=extracted_answer_list)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # compute nonzero scores
                    score_list = []
                    for i_ in range(batch.batch['input_ids'].shape[0]):
                        score_list.append(batch.batch['token_level_scores'][i_].sum().item())

                    metrics.update(self.compute_final_reward_distribution_metrics(score_list))

                    if 'prob' in self.config.reward_model.reward_manager:
                        escape_keys = []
                        if self.config.actor_rollout_ref.actor.get('use_sft_loss', False):
                            escape_keys.extend(['ground_truth_mask_pr', 'old_log_probs_pr']) # For SFT loss
                        batch_keys_rm = [batch_k for batch_k in batch.batch.keys() if (batch_k.endswith('_pr') and batch_k not in escape_keys)]
                        print(f"{batch_keys_rm=}")
                        batch.pop(batch_keys=batch_keys_rm)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch) 
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    if 'prob' in self.config.reward_model.reward_manager:
                        batch_keys_rm = [batch_k for batch_k in batch.batch.keys() if batch_k.endswith('_pr')]
                        if len(batch_keys_rm) != 0:
                            batch.pop(batch_keys=batch_keys_rm)
                    
                    # update score_std_list
                    if config_data_filter_accuracy or config_data_filter_truncated:
                        if self.config.data.get('filter_mode', 'default') == 'EMA':
                            if self.global_steps >= self.config.data.filter_ema_start_step \
                               and self.global_steps < self.config.data.filter_start_step:
                                # score_std_list[-1].extend([batch[i_].batch['final_reward_stds'].item() for i_ in range(len(batch))])
                                score_std_list[-1].extend([batch[i_].batch['final_reward_stds'].item() for i_ in range(len(batch)) if (not math.isnan(batch[i_].batch['final_reward_stds'].item()) and not  math.isinf(batch[i_].batch['final_reward_stds'].item()))])

                                print(f"At the end of the {self.global_steps=}: {len(score_std_list)=} {len(score_std_list[-1])=}")

                            if self.global_steps >= self.config.data.filter_ema_start_step:
                                score_std_list[-1] = sum(score_std_list[-1]) / len(score_std_list[-1])




                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                # acc_reward = acc_tensor.sum(-1)
                scoreB = scoreB_tensor.sum(-1)
                scoreA = scoreA_tensor.sum(-1)
                format_reward = format_reward_tensor.sum(-1)
                metrics.update({# reward
                    'critic/scoreB/mean':
                        torch.mean(scoreB).detach().item(),
                    'critic/scoreB/max':
                        torch.max(scoreB).detach().item(),
                    'critic/scoreB/min':
                        torch.min(scoreB).detach().item(),
                    'critic/scoreA/mean':
                        torch.mean(scoreA).detach().item(),
                    'critic/scoreA/max':
                        torch.max(scoreA).detach().item(),
                    'critic/scoreA/min':
                        torch.min(scoreA).detach().item(),
                    'critic/format_rewards/mean':
                        torch.mean(format_reward).detach().item(),
                    'critic/format_rewards/max':
                        torch.max(format_reward).detach().item(),
                    'critic/format_rewards/min':
                        torch.min(format_reward).detach().item(),
                })
                if 'mix' in self.config.reward_model.reward_manager:
                    metrics.update({
                        "critic/vr_score/mean":
                            torch.mean(exact_tensor.sum(-1).float()).detach().item(),
                        "critic/pr_scoreB/mean":
                            torch.mean(pr_scoreB_tensor.sum(-1).float()).detach().item(),
                        "critic/pr_scoreB/max":
                            torch.max(pr_scoreB_tensor.sum(-1).float()).detach().item(),
                        "critic/pr_scoreB/min":
                            torch.min(pr_scoreB_tensor.sum(-1).float()).detach().item(),
                        'critic/scoreA/mean':
                            torch.mean(pr_scoreA_tensor.sum(-1).float()).detach().item(),
                        'critic/scoreA/max':
                            torch.max(pr_scoreA_tensor.sum(-1).float()).detach().item(),
                        'critic/scoreA/min':
                            torch.min(pr_scoreA_tensor.sum(-1).float()).detach().item(),
                        'critic/all_correct_rate':
                            torch.mean((straightA_tensor[:,0] == 1.).float().mean(-1)).detach().item(),
                        'critic/all_wrong_rate':
                            torch.mean((straightA_tensor[:,0] == -1.).float().mean(-1)).detach().item(),
                    })

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(self.compute_scoreB_by_data_source_metrics(batch=batch, scoreB_tensor=scoreB_tensor, name='scoreB'))
                metrics.update(self.compute_scoreB_by_data_source_metrics(batch=batch, scoreB_tensor=scoreA_tensor, name='scoreA'))
                metrics.update(self.compute_std_per_group_by_data_source_metrics(batch=batch, reward_tensor=reward_tensor, name='std_per_group'))


                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return


    def collate_fn(self, data_list: list[dict]) -> dict:
        tensors = {}
        non_tensors = {}

        for data in data_list:
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    if key not in tensors:
                        tensors[key] = []
                    tensors[key].append(val)
                else:
                    if key not in non_tensors:
                        non_tensors[key] = []
                    non_tensors[key].append(val)

        for key, val in tensors.items():
            tensors[key] = torch.stack(val, dim=0)

        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)

        output = {}
        output.update(tensors)
        output.update(non_tensors)
        return output
    
    def count_pad_tokens(self, s, pad_token_str):
        # Count the number of pad tokens on the left
        left_count = 0
        while s.startswith(pad_token_str):
            left_count += 1
            s = s[len(pad_token_str):]
        
        # Count the number of pad tokens on the right
        right_count = 0
        while s.endswith(pad_token_str):
            right_count += 1
            s = s[:-len(pad_token_str)]
        
        return left_count, right_count

    def get_scoreA(self, data):
        batch_input_ids = data.batch['input_ids'] # [256, 512]
        pad_token_str = self.tokenizer.pad_token
        eos_token_str = self.tokenizer.eos_token
        max_prompt_length, max_response_length = self.config.data.max_prompt_length, self.config.data.max_response_length
        data_list = []
        prompt_str_list, ground_truth_list = [], []
        for i in range(len(batch_input_ids)):
            data_item = data[i]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            prompt_str = self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=False)
            new_text = prompt_str + ' ' + ground_truth + ' ' + eos_token_str
            # new_text_rmpad = new_text.replace(self.tokenizer.pad_token, '')
            new_text_rmpad = replace_left_and_right_str(new_text, self.tokenizer.pad_token)
            if not new_text_rmpad.endswith(eos_token_str): # For a base model, the eos_token_str is the same as pad_token_str
                new_text_rmpad += eos_token_str
            outputs = self.tokenizer(new_text_rmpad, return_tensors='pt', add_special_tokens=False)
            input_ids = outputs['input_ids']
            attention_mask = outputs['attention_mask']
            if '<｜Assistant｜><think>' in self.tokenizer.chat_template:
                sep_str = '<｜Assistant｜><think>' + '\n'
            elif '<｜Assistant｜>' in self.tokenizer.chat_template:
                sep_str = '<｜Assistant｜>' + '\n'
            elif 'assistant<|end_header_id|>' in self.tokenizer.chat_template: # llama 3.1 8b Instruct
                sep_str = 'assistant<|end_header_id|>\n' + '\n'
            elif '<start_of_turn>model' in self.tokenizer.chat_template:
                sep_str = '<start_of_turn>model' + '\n'
            else:
                sep_str = '<|im_start|>assistant' + '\n'
            pos = self.locate_substring_tokens(new_text_rmpad, sep_str, self.tokenizer)

            prompts = input_ids[:, :pos[-1] + 1]
            responses = input_ids[:, pos[-1] + 1:]

            pos_gt = self.locate_substring_tokens(new_text_rmpad, ground_truth, self.tokenizer, ignore_end_text=eos_token_str) # list
            # Note that if GT is empty, this will report errors.
            ground_truth_ids = input_ids[:, pos_gt[0]:pos_gt[-1] + 1]
            start = (pos_gt[0]) - (pos[-1] + 1)


            # Pad prompts and responses for future packing
            left_pad_tuple = (max_prompt_length- prompts.shape[-1], 0)
            right_pad_tuple = (0, max_response_length - responses.shape[-1])

            prompts = F.pad(prompts, left_pad_tuple, 'constant', self.tokenizer.pad_token_id) # pad to be max_length before collate_fn
            responses = F.pad(responses, right_pad_tuple, 'constant', self.tokenizer.pad_token_id) # pad to be max_response_length before collate_fn

            input_ids = torch.cat([prompts, responses], dim=-1)

            # pad right first
            position_ids = compute_position_id_with_mask(F.pad(attention_mask, right_pad_tuple, 'constant', 1))
            attention_mask = F.pad(attention_mask, right_pad_tuple, 'constant', 0)
            # then pad left
            attention_mask = F.pad(attention_mask, left_pad_tuple, 'constant', 0)
            position_ids = F.pad(position_ids, left_pad_tuple, 'constant', 0)

            ground_truth_mask = torch.zeros_like(responses)
            ground_truth_mask[:, start:start + ground_truth_ids.shape[-1]] = 1 # Suppose the response is <think> ABC </think> <answer> DEF </answer>. Then the mask is on " DEF ".



            row_dict = {
                'prompts': prompts[0],
                'responses': responses[0],
                'input_ids': input_ids[0],
                'attention_mask': attention_mask[0],
                'position_ids': position_ids[0],
                'ground_truth_mask': ground_truth_mask[0],
            }

            # prompt_str_list.append(prompt_str.replace(pad_token_str, ''))
            prompt_str_list.append(replace_left_and_right_str(prompt_str, pad_token_str))
            ground_truth_list.append(ground_truth)
            data_list.append(row_dict)

        data_new: DataProto = DataProto.from_single_dict(self.collate_fn(data_list))
        old_log_probs = self.actor_rollout_wg.compute_log_prob(data_new)['old_log_probs'].batch
        scoreAs_list = []
        old_log_probs_in_gt_list = []
        for i in range(len(batch_input_ids)):
            ground_truth_mask = data_new[i].batch['ground_truth_mask']
            old_log_prob = old_log_probs[i]

            old_log_probs_in_gt = old_log_prob[ground_truth_mask.bool()]
            if self.config.reward_model.get('compute_score_name', None) == 'mean_exp_log_softmax':
                scoreA = torch.mean(torch.exp(old_log_probs_in_gt)).item()
            # mean log probs
            elif self.config.reward_model.get('compute_score_name', None) == 'mean_log_softmax':
                scoreA = torch.mean(old_log_probs_in_gt).item()
            # product of probs
            elif self.config.reward_model.get('compute_score_name', None) == 'exp_sum_log_softmax':
                scoreA = torch.exp(torch.sum(old_log_probs_in_gt)).item()
            # root of the product of probs
            elif self.config.reward_model.get('compute_score_name', None) == 'exp_mean_log_softmax':
                scoreA = torch.exp(torch.mean(old_log_probs_in_gt)).item() 
            else:
                raise ValueError
            scoreAs_list.append(scoreA)
            old_log_probs_in_gt_list.append(old_log_prob[ground_truth_mask.bool()])

        return scoreAs_list, prompt_str_list, ground_truth_list, old_log_probs_in_gt_list




    def replace_answer_with_gt_batch(self, gen_ids_batch, gen_response_batch, ground_truth_batch,
                                     prompts_batch_shape, start_think, end_think, eos_token_str,
                                     pad_token_str, start_answer, end_answer, max_length, suffix,
                                     other_answer=False):
        """
        批处理版本的 replace_answer_with_gt 方法

        Args:
            gen_ids_batch: tensor of shape [batch_size, seq_len]
            gen_response_batch: tensor of shape [batch_size, response_len]
            ground_truth_batch: list of strings, length = batch_size
            其他参数与原方法相同

        Returns:
            batch_row_dict: 包含所有批次数据的字典
        """
        batch_size = len(gen_ids_batch)

        # 批量解码文本
        gen_texts = self.tokenizer.batch_decode(gen_ids_batch, skip_special_tokens=False)
        gen_response_texts = self.tokenizer.batch_decode(gen_response_batch, skip_special_tokens=False)

        # 批量处理文本清理
        gen_texts_rmpad = [replace_left_and_right_str(text, pad_token_str) for text in gen_texts]
        gen_response_texts_rmpad = [replace_left_and_right_str(text, pad_token_str) for text in gen_response_texts]

        # 确保所有文本都以 eos_token 结尾
        for i in range(batch_size):
            if not gen_texts_rmpad[i].endswith(eos_token_str): # not in gen_texts_rmpad[i]:
                gen_texts_rmpad[i] += eos_token_str
            # if eos_token_str not in gen_response_texts_rmpad[i]:
            if not gen_response_texts_rmpad[i].endswith(eos_token_str):
                gen_response_texts_rmpad[i] += eos_token_str

        # 批量验证格式和构建新文本
        new_texts = []
        valid_flags = []

        for i in range(batch_size):
            gen_text_rmpad = gen_texts_rmpad[i]
            gen_response_text_rmpad = gen_response_texts_rmpad[i]
            ground_truth = ground_truth_batch[i]

            if self.config.reward_model.get('format_mode', 'R1') == 'R1':
                start_think_count = gen_response_text_rmpad.count(start_think)
                end_think_count = gen_response_text_rmpad.count(end_think)
            middle_content, leading_whitespace, trailing_whitespace = ' ', ' ', ' '

            start_answer_tag = '<answer>'
            start_answer_count = gen_response_text_rmpad.count(start_answer_tag)
            if self.config.reward_model.get('format_mode', 'R1') == 'R1':
                pattern = r'^.*' + start_think + r'.*' + end_think + r'.*' + start_answer_tag + r'.*$'
            elif self.config.reward_model.get('format_mode', 'R1') == 'R1_nothink':
                pattern = r'^.*' + start_answer_tag + r'.*$'
            valid_flag = (
                    start_answer_count == 1 and
                    (re.fullmatch(pattern, gen_response_text_rmpad, re.DOTALL) is not None)
            )
            if self.config.reward_model.get('format_mode', 'R1') == 'R1':
                valid_flag = (
                    valid_flag and 
                    start_think_count == 1 and
                    end_think_count == 1
                )

            if valid_flag:
                if self.config.reward_model.get('format_mode', 'R1') == 'R1':
                    middle_content = gen_response_text_rmpad.split(end_think)[1].split(start_answer_tag)[0]
                answer_section = gen_response_text_rmpad.split(start_answer_tag)[1]

                if not answer_section.strip():
                    valid_flag = False
                else:
                    leading_whitespace = ''
                    for char in answer_section:
                        if char in [' ', '\n', '\t', '\r']:
                            leading_whitespace += char
                        else:
                            break
                    if self.config.reward_model.get("gt_tokens_one_more", False):
                        match = re.search('(\s*)</answer>', answer_section)
                        if match:
                            trailing_whitespace = match.group(1)

            # 处理空白字符
            if not self.config.reward_model.get("allow_empty_leading_whitespaces", False):
                leading_whitespace = leading_whitespace if leading_whitespace != '' else ' '
                leading_whitespace = '' if other_answer else leading_whitespace

            if self.config.reward_model.get("gt_tokens_one_more", False):
                pass
            else:
                trailing_whitespace = trailing_whitespace if trailing_whitespace != '' else ' '
                trailing_whitespace = '' if other_answer else trailing_whitespace

            # 构建新文本
            if valid_flag:
                if self.config.reward_model.get('format_mode', 'R1') == 'R1':
                    new_text = (end_think.join(gen_text_rmpad.split(end_think)[:-1]) +
                                end_think + middle_content + start_answer +
                                leading_whitespace + ground_truth + trailing_whitespace +
                                end_answer + eos_token_str)
                elif self.config.reward_model.get('format_mode', 'R1') == 'R1_nothink':
                    new_text = (start_answer.join(gen_text_rmpad.split(start_answer)[:-1]) + 
                                start_answer +
                                leading_whitespace + ground_truth + trailing_whitespace + 
                                end_answer + eos_token_str)
            else:
                if self.config.reward_model.get('format_mode', 'R1') == 'R1':
                    end_text = (end_think + middle_content + start_answer +
                                leading_whitespace + ground_truth + trailing_whitespace +
                                end_answer + eos_token_str)
                elif self.config.reward_model.get('format_mode', 'R1') == 'R1_nothink':
                    end_text = (start_answer + 
                                leading_whitespace + ground_truth + trailing_whitespace + 
                                end_answer + eos_token_str)

                end_text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(end_text))
                new_text = (replace_left_and_right_str(self.tokenizer.decode(gen_ids_batch[i][:-len(end_text_ids) - 5],
                                                  skip_special_tokens=False), self.tokenizer.pad_token) +
                            end_text)

            new_texts.append(new_text)
            valid_flags.append(valid_flag)


        # 批量 tokenize 新文本
        batch_input_data = self.tokenizer(new_texts, return_tensors='pt',
                                          add_special_tokens=False, truncation=True,
                                          max_length=max_length + self.config.data.max_response_length)

        batch_input_ids = batch_input_data['input_ids']
        batch_attention_mask = batch_input_data['attention_mask']

        # 批量定位 start_answer 和 ground_truth 位置
        pos_startanswer_batch = self.batch_locate_substring_tokens(new_texts, start_answer)
        pos_gt_batch = self.batch_locate_substring_tokens(new_texts, ground_truth_batch,
                                                     ignore_end_text=end_answer + eos_token_str)

        # 处理 gt_tokens_one_more 配置
        if self.config.reward_model.get("gt_tokens_one_more", False):
            for pos_gt in pos_gt_batch:
                if pos_gt:
                    pos_gt.append(pos_gt[-1] + 1)

        # 批量构建最终结果
        batch_results = {
            f'prompts{suffix}': [],
            f'responses{suffix}': [],
            f'input_ids{suffix}': [],
            f'attention_mask{suffix}': [],
            f'position_ids{suffix}': [],
            f'ground_truth_mask{suffix}': [],
        }

        for i in range(batch_size):
            if not valid_flags[i] or torch.sum(batch_input_ids[i] != self.tokenizer.pad_token_id).item() > max_length:
                valid_flags[i] = False

            pos_startanswer = pos_startanswer_batch[i]
            pos_gt = pos_gt_batch[i]

            if pos_startanswer and pos_gt and len(pos_startanswer) > 0 and len(pos_gt) > 0:
                start_pos = pos_startanswer[0]
                gt_start = pos_gt[0]
                gt_end = pos_gt[-1]

                prompts = batch_input_ids[i:i + 1, :start_pos]
                responses = batch_input_ids[i:i + 1, start_pos:]

                ground_truth_ids = batch_input_ids[i:i + 1, gt_start:gt_end + 1]
                start = gt_start - start_pos

                # Padding
                left_pad_tuple = (max_length - prompts.shape[-1], 0)
                right_pad_tuple = (0, self.config.data.max_response_length - responses.shape[-1])

                prompts = F.pad(prompts, left_pad_tuple, 'constant', self.tokenizer.pad_token_id)
                responses = F.pad(responses, right_pad_tuple, 'constant', self.tokenizer.pad_token_id)

                input_ids = torch.cat([prompts, responses], dim=-1)

                # 处理 attention_mask 和 position_ids
                attention_mask = batch_attention_mask[i:i + 1]
                position_ids = compute_position_id_with_mask(F.pad(attention_mask, right_pad_tuple, 'constant', 1))
                attention_mask = F.pad(attention_mask, right_pad_tuple, 'constant', 0)
                attention_mask = F.pad(attention_mask, left_pad_tuple, 'constant', 0)
                position_ids = F.pad(position_ids, left_pad_tuple, 'constant', 0)

                # 处理 ground_truth_mask
                ground_truth_mask = torch.zeros_like(responses)
                if valid_flags[i]:
                    ground_truth_mask[:, start:start + ground_truth_ids.shape[-1]] = 1

            else:
                # 创建默认值
                prompts = torch.zeros(1, max_length, dtype=torch.long)
                responses = torch.zeros(1, self.config.data.max_response_length, dtype=torch.long)
                input_ids = torch.cat([prompts, responses], dim=-1)
                attention_mask = torch.zeros_like(input_ids)
                position_ids = torch.zeros_like(input_ids)
                ground_truth_mask = torch.zeros_like(responses)

            # 添加到批次结果
            batch_results[f'prompts{suffix}'].append(prompts[0])
            batch_results[f'responses{suffix}'].append(responses[0])
            batch_results[f'input_ids{suffix}'].append(input_ids[0])
            batch_results[f'attention_mask{suffix}'].append(attention_mask[0])
            batch_results[f'position_ids{suffix}'].append(position_ids[0])
            batch_results[f'ground_truth_mask{suffix}'].append(ground_truth_mask[0])

        # 转换为张量
        for key in batch_results:
            batch_results[key] = torch.stack(batch_results[key])

        return batch_results

    def batch_locate_substring_tokens(self, full_strings, substrings, ignore_end_text=None):
        """
        Locates the token IDs and positions corresponding to a substring in a full string.
        Args:
            full_string (List[str]): The full string to tokenize.
            substring (List[str]): The substring to locate in the full string.
            tokenizer_name (List[str]): The name of the tokenizer to use (default is "gpt2").
        """
        # Tokenize the full string and get byte-level offsets
        batch_encodings = self.tokenizer(full_strings, return_offsets_mapping=True, add_special_tokens=False)
        batch_offsets = batch_encodings["offset_mapping"]  # List of (start, end) byte positions for each token
        # Find the byte-level start and end positions of the substring in the full string
        batch_matching_token_indices = []
        for string_idx in range(len(full_strings)):
            full_string = full_strings[string_idx]
            if isinstance(substrings, str):
                substring = substrings
            else:
                substring = substrings[string_idx]
            offsets = batch_offsets[string_idx]
            if ignore_end_text is not None:
                assert full_string.endswith(
                    ignore_end_text), f"{full_string=} given but {ignore_end_text=} not in the end of the full string"
                sub_start = full_string[:-len(ignore_end_text)].rfind(substring)
            else:
                sub_start = full_string.rfind(substring)
            if sub_start == -1:
                print(f"{full_string=}")
                raise ValueError(f"Substring `{substring}` not found in the full string.")
            sub_end = sub_start + len(substring)
            # Locate the tokens that overlap with the substring's byte range
            matching_token_indices = [
                i for i, (start, end) in enumerate(offsets)
                if start < sub_end and end > sub_start
            ]
            batch_matching_token_indices.append(matching_token_indices)
        return batch_matching_token_indices

    def construct_new_batch_optimized(self, gen_batch_output, ground_truth_list,
                                      start_think='<think>', end_think='</think>',
                                      start_answer='<answer>', end_answer='</answer>',
                                      suffix='_pr'):
        """
        优化后的批处理版本的 construct_new_batch
        """
        self.tokenizer = TokenizerWrapper(tokenizer=self.tokenizer)

        gen_ids = gen_batch_output.batch['input_ids']  # prompt + response
        gen_responses = gen_batch_output.batch['responses']  # response only

        pad_token_str = self.tokenizer.pad_token
        eos_token_str = self.tokenizer.eos_token
        max_length = self.config.data.max_prompt_length + self.config.data.max_response_length

        # 调用批处理版本的方法
        batch_results = self.replace_answer_with_gt_batch(
            gen_ids,
            gen_responses,
            ground_truth_list,
            gen_batch_output.batch['prompts'].shape[-1],
            start_think,
            end_think,
            eos_token_str,
            pad_token_str,
            start_answer,
            end_answer,
            max_length,
            suffix
        )

        gen_batch_output: DataProto = DataProto.from_single_dict(batch_results)
        self.tokenizer = self.tokenizer.tokenizer
        return gen_batch_output

    def locate_substring_tokens(self, full_string, substring, tokenizer, ignore_end_text=None):
        """
        Locates the token IDs and positions corresponding to a substring in a full string.

        Args:
            full_string (str): The full string to tokenize.
            substring (str): The substring to locate in the full string.
            tokenizer_name (str): The name of the tokenizer to use (default is "gpt2").
        """
        # Tokenize the full string and get byte-level offsets
        encoding = tokenizer(full_string, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding["offset_mapping"]  # List of (start, end) byte positions for each token

        # Find the byte-level start and end positions of the substring in the full string
        if ignore_end_text is not None:
            assert full_string.endswith(ignore_end_text), f"{full_string=} given but {ignore_end_text=} not in the end of the full string"
            sub_start = full_string[:-len(ignore_end_text)].rfind(substring)
        else:
            sub_start = full_string.rfind(substring)
        if sub_start == -1:
            print(f"{full_string=}")
            raise ValueError(f"Substring `{substring}` not found in the full string.")
        sub_end = sub_start + len(substring)

        # Locate the tokens that overlap with the substring's byte range
        matching_token_indices = [
            i for i, (start, end) in enumerate(offsets)
            if start < sub_end and end > sub_start
        ]

        return matching_token_indices

    def compute_final_reward_distribution_metrics(self, score_list):
        # Initialize a list to store the counts for each bin (11 bins now)
        bin_counts = [0] * 12  # 10 bins for [0, 1.0) + 1 bin for out-of-range values

        # Iterate over the scores in score_list
        num_zero_score, num_one_score = 0, 0
        for score in score_list:
            if score == 0:
                num_zero_score += 1
            elif score == 1:
                num_one_score += 1
            elif 0 < score < 1.0:  # Check if the score is within [0, 1.0)
                bin_index = int(score * 10)  # Determine the bin index
                bin_counts[bin_index] += 1
            elif score < 0:
                bin_counts[10] += 1  # The 11th bin is for out-of-range values
            elif score > 1.0: # score > 1.0
                # Handle out-of-range values (negative or >= 1.0)
                bin_counts[11] += 1  # The 11th bin is for out-of-range values

        # Create a metrics dictionary to store the bin counts
        metrics = {}
        metrics['final_reward_dist/final_reward<0'] = bin_counts[10] / len(score_list)
        metrics['final_reward_dist/final_reward=0'] = num_zero_score / len(score_list)
        for i in range(10):
            metrics[f'final_reward_dist/final_reward_in_{i/10:.1f}_to_{(i+1)/10:.1f}'] = bin_counts[i] / len(score_list)
        metrics['final_reward_dist/final_reward=1'] = num_one_score / len(score_list)
        metrics['final_reward_dist/final_reward>1'] = bin_counts[11] / len(score_list)


        return metrics



    @staticmethod
    def compute_reward_distributions_by_group(reward_tensor, batch):
        """
        Compute reward distributions for each group based on `uid`.

        Args:
            reward_tensor (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing rewards.
            batch (list): A list of batch elements where each element has a `non_tensor_batch` attribute
                        containing the `uid` for grouping.

        Returns:
            dict: A dictionary mapping each `uid` to its reward distribution statistics (mean, std, min, max, and raw rewards).
        """
        # Step 1: Collect rewards for each group
        uid_to_rewards = defaultdict(list)

        for i in range(reward_tensor.shape[0]):
            uid = batch[i].non_tensor_batch['uid']
            # Since each reward tensor contains at most 1 nonzero value, we use `sum()` to extract that
            rewards = reward_tensor[i].sum().item()  # Sum of rewards for the current sequence.
            uid_to_rewards[uid].append(rewards)

        # Step 2: Compute the distribution for each group
        uid_to_distribution = {}

        for uid, rewards in uid_to_rewards.items():
            rewards_tensor = torch.tensor(rewards)
            mean = rewards_tensor.mean().item()
            std = rewards_tensor.std().item()
            min_val = rewards_tensor.min().item()
            max_val = rewards_tensor.max().item()
            
            uid_to_distribution[uid] = {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'gap': max_val - min_val,
                'rewards': rewards  # Optionally, keep the raw rewards for further analysis
            }

        return uid_to_distribution
    
    @staticmethod
    def compute_row_mean(responses, attention_mask):
        row_mean = []
        response_mask = attention_mask[:, -responses.shape[1]:]
        for i in range(responses.size(0)):
            # Get non-padded tokens for the current row
            non_padded_tokens = responses[i][response_mask[i] == 1]
            # Compute std for non-padded tokens
            if non_padded_tokens.numel() > 0:  # Ensure there are non-padded tokens
                row_mean.append(non_padded_tokens.mean())
            else:
                row_mean.append(torch.tensor(0.0))  # If all tokens are padded, std is 0
        return torch.stack(row_mean)
    # Function to compute row-wise std, ignoring padded values


    def sample_batch_data(self, batch, reward_tensor, scoreA_tensor, scoreB_tensor, advantage_list, entropy_list, format_reward_tensor, extracted_answer_list, N,
                          exact_tensor=None, pr_scoreB_tensor=None, pr_scoreA_tensor=None):
        # Extract all unique uids from the batch
        unique_uids = list(set(item.non_tensor_batch['uid'] for item in batch))
        
        # Ensure N is not greater than the number of unique uids
        N = min(N, len(unique_uids))
        
        # Randomly select N unique uids
        selected_uids = random.sample(unique_uids, N)
        

        # Initialize the log dictionary
        log_to_wandb = {
            "scores": [],
            "scoreAs": [],
            "scoreBs": [],
            "advantages": [],
            "entropies": [],
            "format_rewards": [],
            "data_sources": [],
            "sequences": [],
            "extracted_answers": [],
            "ground_truths": [],
            "final_reward_stds": [],
        }
        if 'mix' in self.config.reward_model.reward_manager:
            log_to_wandb.update({
                "vr_scores": [],
                "pr_scoreAs": [],
                "pr_scoreBs": [],
            })
        
        # Iterate over the batch and collect data for the selected uids
        for i_ in range(len(batch)):
            if batch[i_].non_tensor_batch['uid'] in selected_uids:
                score_ = reward_tensor[i_].sum().item()
                scoreA_ = scoreA_tensor[i_].sum().item()
                scoreB_ = scoreB_tensor[i_].sum().item()
                advantage_ = advantage_list[i_]
                entropy_ = entropy_list[i_]
                format_reward_ = format_reward_tensor[i_].sum().item()
                data_source_ = batch[i_].non_tensor_batch['data_source']
                sequence_ = self.tokenizer.decode(batch.batch['input_ids'][i_][batch.batch['attention_mask'][i_].bool()], skip_special_tokens=False)
                extracted_answer_ = extracted_answer_list[i_]
                ground_truth_ = batch[i_].non_tensor_batch['reward_model']['ground_truth']
                final_reward_std_ = batch[i_].batch['final_reward_stds'].item()
                
                # Append the data to the log dictionary
                log_to_wandb['scores'].append(score_)
                log_to_wandb['scoreAs'].append(scoreA_)
                log_to_wandb['scoreBs'].append(scoreB_)
                log_to_wandb['advantages'].append(advantage_)
                log_to_wandb['entropies'].append(entropy_)
                log_to_wandb['format_rewards'].append(format_reward_)
                log_to_wandb['data_sources'].append(data_source_)
                log_to_wandb['sequences'].append(sequence_)
                log_to_wandb['extracted_answers'].append(extracted_answer_)
                log_to_wandb['ground_truths'].append(ground_truth_)
                log_to_wandb['final_reward_stds'].append(final_reward_std_)

                if 'mix' in self.config.reward_model.reward_manager:
                    vr_score_ = exact_tensor[i_].sum().item()
                    pr_scoreA_ = pr_scoreA_tensor[i_].sum().item()
                    pr_scoreB_ = pr_scoreB_tensor[i_].sum().item()
                    log_to_wandb['vr_scores'].append(vr_score_)
                    log_to_wandb['pr_scoreAs'].append(pr_scoreA_)
                    log_to_wandb['pr_scoreBs'].append(pr_scoreB_)

                selected_uids.remove(batch[i_].non_tensor_batch['uid'])
        
        return log_to_wandb



    def log_train_generations(self, batch, reward_tensor, scoreA_tensor, scoreB_tensor, advantage_list, entropy_list, format_reward_tensor, extracted_answer_list,
                              exact_tensor=None, pr_scoreB_tensor=None, pr_scoreA_tensor=None):
        uid_to_items = {}
        for i, item in enumerate(batch):
            uid = item.non_tensor_batch['uid']
            if uid not in uid_to_items:
                uid_to_items[uid] = []
            uid_to_items[uid].append(i)

        # Process items grouped by uid
        records = []
        for uid, item_indices in uid_to_items.items():
            for i in item_indices:
                record = {
                    'uid': uid,
                    'score': reward_tensor[i].sum().item(),
                    'scoreA': scoreA_tensor[i].sum().item(),
                    "final_reward_std": batch[i].batch['final_reward_stds'].item(),
                    'scoreB': scoreB_tensor[i].sum().item(),
                    'advantage': advantage_list[i],
                    'entropy': entropy_list[i],
                    'format_reward': format_reward_tensor[i].sum().item(),
                    'data_source': batch[i].non_tensor_batch['data_source'],
                    'sequence': self.tokenizer.decode(
                        batch.batch['input_ids'][i][batch.batch['attention_mask'][i].bool()], 
                        skip_special_tokens=False),
                    'extracted_answer': extracted_answer_list[i],
                    'ground_truth': batch[i].non_tensor_batch['reward_model']['ground_truth']
                }
                if 'mix' in self.config.reward_model.reward_manager:
                    record.update({
                        'vr_score': exact_tensor[i].sum().item(),
                        'pr_scoreA': pr_scoreA_tensor[i].sum().item(),
                        'pr_scoreB': pr_scoreB_tensor[i].sum().item(),
                    })
                records.append(record)

        df = pd.DataFrame(records)
        df.sort_values('uid', inplace=True)

        # Create directory if it doesn't exist
        save_path = f"data/logs/train_generations/{self.config.trainer.experiment_name}_step{self.global_steps}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        try:
            print(f"{save_path=}")
            df.to_csv(save_path, index=False, escapechar='\\')
            df_excel = pd.read_csv(save_path, encoding='utf-8')
            excel_path = save_path.replace('.csv', '.xlsx')
            print(f"{excel_path=}")
            df_excel.to_excel(excel_path, index=False, engine='xlsxwriter')
        except Exception as e: # we meet errors like _csv.Error: need to escape, but no escapechar set
            print(f"Error saving train_generations: {e}")


    @staticmethod
    def compute_scoreB_by_data_source_metrics(batch, scoreB_tensor, name='scoreB'):
        # Initialize a dictionary to store scoreB_reward values for each data_source
        scoreB_by_source = defaultdict(list)
        
        # Iterate over the batch and collect scoreB_reward for each data_source
        for i in range(len(batch)):
            data_source = batch[i].non_tensor_batch['data_source']
            scoreB = scoreB_tensor[i].sum(-1).detach().item()
            scoreB_by_source[data_source].append(scoreB)
        
        # Calculate mean, max, and min for each data_source and format the keys
        metrics_by_source = {}
        for data_source, rewards in scoreB_by_source.items():
            rewards_tensor = torch.tensor(rewards)
            metrics_by_source[f'critic_wrt_data_source/{name}/{data_source}/mean'] = torch.mean(rewards_tensor).item()
        
        return metrics_by_source

    def compute_std_per_group_by_data_source_metrics(self, batch, reward_tensor, name='std_per_group'):
        # First compute the reward distributions to get std_per_group for each item
        uid_to_distribution = self.compute_reward_distributions_by_group(reward_tensor, batch)
        
        # Initialize a dictionary to store std_per_group values for each data_source
        std_per_group_by_source = defaultdict(list)
        
        # Iterate over the batch and collect std_per_group for each data_source
        for i in range(len(batch)):
            uid = batch[i].non_tensor_batch['uid']
            data_source = batch[i].non_tensor_batch['data_source']
            
            # Get the std for this item from the precomputed distributions
            std = uid_to_distribution[uid]['std']
            std_per_group_by_source[data_source].append(std)
        
        # Calculate mean for each data_source and format the keys
        metrics_by_source = {}
        for data_source, std_values in std_per_group_by_source.items():
            std_tensor = torch.tensor(std_values)
            metrics_by_source[f'critic_wrt_data_source/{name}/{data_source}/mean'] = torch.mean(std_tensor).item()
        
        return metrics_by_source


    def compute_promptgt2scoreA(self, epoch: int) -> None:
        """
        Processes and logs the distribution of scoreA for the given epoch.
        
        Args:
            epoch (int): The current epoch number.
        """
        # Check if probabilistic reward is enabled in the configuration
        if 'prob' not in self.config.reward_model.reward_manager:
            return

        # Set the seed for reproducibility
        current_seed = self.config.data.get('seed', 1) if epoch == 0 else random.randint(0, 2**32 - 1)
        if self.config.data.shuffle:
            self.train_dataloader.sampler.generator.manual_seed(current_seed)

        save_path = os.path.join(self.config.trainer.default_local_dir, 'promptgt2scoreA.json')

        promptgt2scoreA = {}
        if os.path.exists(save_path):
            promptgt2scoreA = load_from_json(save_path)
        else:
            scoreA_list = []

            total_train_samples = len(self.train_dataloader.dataset)

            train_dataloader = self.train_dataloader_repeat

            for idx, batch_dict in tqdm(enumerate(train_dataloader), total=len(self.train_dataloader) + 4):
                print(f"{idx=} {len(promptgt2scoreA)=} {len(scoreA_list)=}. The goal is {total_train_samples}")
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                scoreAs, prompt_strs, ground_truths, old_log_probs_in_gt_list = self.get_scoreA(batch)
                # Process each item in the batch

                for i in range(len(batch)):
                    prompt = replace_left_and_right_str(self.tokenizer.decode(
                        batch.batch[i]['input_ids'], 
                        skip_special_tokens=False
                    ), self.tokenizer.pad_token)
                    ground_truth = batch[i].non_tensor_batch['reward_model']['ground_truth']
                    key = prompt + '\n\n\n' + ground_truth
                    if key not in promptgt2scoreA:
                        promptgt2scoreA[key] = scoreAs[i]
                        scoreA_list.append(scoreAs[i])
                if idx >= len(self.train_dataloader) + 4:
                    break

            if 'wandb' in self.config.trainer.logger:
                # Log the scoreA distribution to WandB
                self._maybe_log_histogram_to_wandb(
                    scoreA_list, 
                    'figures/scoreA', 
                    'scoreA', 
                    'Score A Distribution'
                )

            save_to_json(promptgt2scoreA, save_path)



        # Reset the seed to ensure consistent data order for training
        if self.config.data.shuffle:
            self.train_dataloader.sampler.generator.manual_seed(current_seed)

        print(f"{len(promptgt2scoreA)=}")
        return promptgt2scoreA

    def save_rollout(self, batch, ground_truth_list, this_index_list=None):
        print(f"{batch=}")

        prompts = self.tokenizer.batch_decode(
            batch.batch['prompts'], 
            skip_special_tokens=True
        )
        responses = self.tokenizer.batch_decode(
            batch.batch['responses'], 
            skip_special_tokens=True
        )

        results = []
        for i in range(len(prompts)):
            results.append({
                "Index": i,
                "Prompt": prompts[i],
                "Response": responses[i],
                "ground_truth": ground_truth_list[i],
                "this_index": this_index_list[i],
            })
        
        # Create directory if it doesn't exist
        save_path = f"data/logs/rollout/{self.config.trainer.experiment_name}_step{self.global_steps}.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df = pd.DataFrame(results)
        # Save to CSV
        print(f"{save_path=}")
        df.to_csv(save_path, index=False, escapechar='\\')
        df_excel = pd.read_csv(save_path, encoding='utf-8')
        excel_path = save_path.replace('.csv', '.xlsx')
        print(f"{excel_path=}")
        df_excel.to_excel(excel_path, index=False, engine='xlsxwriter')
        
    def compute_think_answer_length_metrics(self, batch) -> Dict[str, float]:
        """
        Compute token length statistics for text within <think> and <answer> tags.
        If the text does not match the expected pattern, counts are set to 0.
        
        Args:
            batch: Contains response data with 'responses' field of token IDs
            
        Returns:
            Dictionary with mean/max/min token lengths for think and answer sections.
            If no matches exist, all metrics will be 0.
        """
        output_ids = batch.batch['responses']
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        
        think_lengths = []
        answer_lengths = []
        
        if self.config.reward_model.get('format_mode', 'R1') == 'R1':
            pattern = re.compile(r'.*<think>(.*)</think>.*<answer>(.*)</answer>.*', re.DOTALL)
        elif self.config.reward_model.get('format_mode', 'R1') == 'R1_nothink':
            pattern = re.compile(r'(.*)<answer>(.*)</answer>.*', re.DOTALL)
        else:
            raise ValueError
        
        for text in output_texts:
            match = pattern.fullmatch(text)
            
            if match:
                think_text, answer_text = match.groups()
                think_tokens = self.tokenizer.tokenize(think_text.strip())
                answer_tokens = self.tokenizer.tokenize(answer_text.strip())
                
                think_lengths.append(len(think_tokens))
                answer_lengths.append(len(answer_tokens))
            else:
                think_lengths.append(0)
                answer_lengths.append(0)
        
        # Compute statistics (if no entries, np.mean/max/min will return NaN, so we handle that)
        def safe_stats(values: List[int]) -> Dict[str, float]:
            if not values:  # Empty list
                return {"mean": 0.0, "max": 0.0, "min": 0.0}
            return {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "min": float(np.min(values)),
            }
        
        think_stats = safe_stats(think_lengths)
        answer_stats = safe_stats(answer_lengths)
        
        return {
            "think_length/mean": think_stats["mean"],
            "think_length/max": think_stats["max"],
            "think_length/min": think_stats["min"],
            "answer_length/mean": answer_stats["mean"],
            "answer_length/max": answer_stats["max"],
            "answer_length/min": answer_stats["min"],
        }

    @staticmethod
    def calculate_ema(data, ratio):
        """
        Calculate the Exponential Moving Average (EMA) of a list of floats.
        The weights decrease exponentially with a ratio of 0.9 for each previous element.
        The last element has weight 1, the previous 0.9, then 0.9^2, etc.
        
        Args:
            data: List of floats to calculate EMA for
            
        Returns:
            The EMA value as a float
        """
        if not data:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        current_weight = 1.0  # Start with weight 1 for the last element

        print(f'{data=}')
        
        # Iterate from the end to the beginning
        for value in reversed(data):
            weighted_sum += value * current_weight
            total_weight += current_weight
            current_weight *= ratio
        
        return weighted_sum / total_weight
    @staticmethod
    def compute_std_per_group_merics(reward_tensor, batch, prefix):
        uid_to_distribution = RayPPOTrainer.compute_reward_distributions_by_group(reward_tensor, batch)
        std_per_group, gap_per_group, mean_per_group = [], [], []
        for group_idx, (uid, distribution) in enumerate(uid_to_distribution.items()):
            mean_per_group.append(distribution['mean'])
            std_per_group.append(distribution['std'])
            gap_per_group.append(distribution['gap'])
        
        return({
            f"{prefix}/mean_per_group/mean": sum(mean_per_group) / len(mean_per_group),
            f"{prefix}/mean_per_group/max": max(mean_per_group),
            f"{prefix}/mean_per_group/min": min(mean_per_group),
            f"{prefix}/std_per_group/mean": sum(std_per_group) / len(std_per_group),
            f"{prefix}/std_per_group/max": max(std_per_group),
            f"{prefix}/std_per_group/min": min(std_per_group),
        })


    @staticmethod
    def compute_mean_of_metrics(metrics):
        """
        Compute the mean of values in the metrics dictionary where keys follow the pattern
        "critic_during_filter/rewards_{idx}/std_per_group/mean"
        
        Args:
            metrics (dict): Dictionary containing metric values
        
        Returns:
            float: Mean of the matching values
        """
        result = {}
        for prefix, postfix in [('rewards', '/std_per_group/mean'), ('rewards', '/mean_per_group/mean'), ('scoreB', 'mean')]:
            total = 0.0
            count = 0
            
            for key in metrics.keys():
                if key.startswith(f"critic_during_filter/{prefix}_") and key.endswith(postfix):
                    try:
                        # Extract the index part (might need adjustment based on actual key structure)
                        idx_part = key.split(f"critic_during_filter/{prefix}_")[1].split(postfix)[0]
                        # Try to convert to int to verify it's a numeric index
                        int(idx_part)
                        total += metrics[key]
                        count += 1
                    except (IndexError, ValueError):
                        # Skip if the key doesn't match our expected pattern
                        continue
            
            if count == 0:
                print("No matching keys found in the metrics dictionary")
                return {}
                # raise ValueError("No matching keys found in the metrics dictionary")
            
            result[f'critic_during_filter_mean/{prefix}{postfix}'] = total / count
        return result

def replace_left_and_right_str(text, left_str):
    while text.startswith(left_str):
        text = text[len(left_str):]
    while text.endswith(left_str):
        text = text[:-len(left_str)]
    return text

def load_from_json(path):
    with open(path, 'r') as file:
        print(f"Loading from {path}")
        result = json.load(file)
    return result

def save_to_json(result, path):
    with open(path, 'w') as file:
        print(f"We dump to {path}")
        json.dump(result, file)