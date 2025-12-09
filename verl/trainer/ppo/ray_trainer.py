# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
from pandas.core.base import NoNewAttributesMixin
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto # 数据说明已在子文件中进行理解
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.reward_manager import batch

from verl.utils.add_noise import image_augment_from_PIL
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from verl.utils.data_processor import DATA_PROCESSOR_MAP


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]

'''
注意，所有更新用的Advantage和reward都是从这里出的，最后公式中更新的reward就是这个函数安徽的return。
但是如果想自己设计reward等的相关，应该关注compute_reward函数，compute_advantage这个函数是对优势和reward进行后处理的
'''
def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys(): # 如果没有responsemask就计算
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        # ⏰⏰
        advantages, returns = core_algos.compute_grpo_outcome_advantage( # 计算优势和回报
            token_level_rewards=data.batch["token_level_rewards"],# 这里面的return是根据即时奖励计算出未来的累积折扣奖励
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        
        

        # Instantiate the tokenizer and processor.
        
        from verl.utils.fs import copy_to_local
        self.local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        self.trust_remote_code = config.data.get("trust_remote_code", False)
        # Used for multimodal LLM, could be None
        


        self.config = config
        self.reward_fn = reward_fn # 用于计算奖励的函数
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )# 启用评测记录，比如wandb

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        # self.ref_in_actor 是一个布尔值，表示 Reference Policy 的模型权重是否嵌入在 Actor Policy 的 Worker 中。
        # 为什么RefPolicy的权重会被嵌入在ActorPolicy当中呢？因为ref的策略参数是一直不会动的，只有policy的策略会动
        # W_ref=W_0 W_policy=W_0+BA 所以把ref的权重嵌入进去避免重复加载权重
        self.ref_in_actor = ( 
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        ) # 如果用了lora，输入lora的path

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward: # KL散度奖励
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn
        # collate_fn是dataset中的数据转换成tensor形式并且可供模型进行训练使用,‼️他还可以进行动态填充，识别
        # 最长的序列然后用padding token将他们填充到同一长度

        num_workers = self.config.data["dataloader_num_workers"]


        # 这个Statefuldataloader是分布式训练中可以进行检查点恢复的dataloader
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None: # 如果用户指定了训练步骤 允许直接覆盖使用
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # 利用OmegaConf将trainingsteps安全写入actor和critic
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        # 将 rollout/验证样本以 JSONL 格式转储到文件中。
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data( # 将rollout数据记录到磁盘
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=False)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=False)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
    
    

    def _cut_log_rollout_data( # 将rollout数据记录到磁盘
        self, batch: DataProto,  rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        
        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=False)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

        # 2. 准备文件路径
        os.makedirs(rollout_data_dir, exist_ok=True)
        filename = os.path.join(rollout_data_dir, f"{self.global_steps}.jsonl")

        # 3. 构建 JSON Lines 列表
        lines = []
        n = len(inputs)

        for i in range(n):
            # 为每个样本创建只包含 input 和 output 的字典
            entry = {
                "input": inputs[i],
                "output": outputs[i],
            }
            lines.append(json.dumps(entry, ensure_ascii=False))

        # 4. 写入文件
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped prompt and rollout results to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):# 将验证样本的生成结果记录到配置的日志器（如 wandb 或 swanlab）中
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto: # 从原始批次中获取用于生成（generation）的批次数据
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self): # 执行验证（validation）过程，计算奖励模型的分数并记录生成样本
        """Validate the model on the validation dataset."""
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader: # 默认所有的验证样本都参与验证，所以len(val_dataloader) = 1
            test_batch = DataProto.from_single_dict(test_data) # 转化成DataProto格式

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )
            print("\n -------_validate stage -------- \n")
            print("Original Test_batch len:{}\n".format(len(test_batch)))
            # repeat test batch
            # 每个原始验证样本会被重复多少次（根据val_kwargs.n进行重复）
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            print("After Repeated Test_batch len:{}\n".format(len(test_batch)))
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            # 获取输入id，并将对应id的文本解码为输入文本
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"]) # 放入sample 

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ] # 提取真实的groundtruth并用于接下来的评估
            sample_gts.extend(ground_truths) # 放入sample   

            # ._get_gen_batch 将test_batch转换为适合模型进行序列生成的格式
            test_gen_batch = self._get_gen_batch(test_batch) #准备用于模型生成的批次。
            print("Test_gen_batch len:{}\n".format(len(test_gen_batch)))
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # 用 actor worker group来生成序列。这是模型实际进行推理并生成输出的地方。
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size) # 恢复原始长度
            print("After Unpadded Test_output_gen_batch len:{}\n".format(len(test_output_gen_batch)))


            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"] # 存储
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids] # 解码
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            '''
            模型生成序列之后， test_output_gen_batch 包含了模型生成的响应（ responses ）以及其
            他生成相关的输出（例如， log_probs ）。而 test_batch 仍然包含原始的输入提示和一些初始元数据。 
            union 操作将这两部分数据合并到一个 DataProto 对象中。为了保证数据的完整性
            '''
            print("After Union Test_batch len:{}\n".format(len(test_batch)))
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        # 启动所有组的性能分析
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        # 停止所有组的性能分析
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        # 有的attention_mask长度大，有的小，这样的长度大的GPU结束时间比较长，长度小的结束快，GPU就空置了，利用率不高
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
    def _truncate_batch_before(self, gen_cut_batch: DataProto):# 传入的gen_cut_batch形状是batch_size*rollout_n
        import verl.utils.torch_functional as verl_F
        from verl.utils import hf_processor
        # from verl.utils.dataset.vision_utils import process_image
        data_processor = hf_processor(self.local_path, trust_remote_code=self.trust_remote_code, use_fast=True)

        prompt_token_ids = gen_cut_batch.batch["input_ids"] # gen_cut_batch.batch.size()=([1024]) gen.batch是个字典
        # batch的key['attention_mask', 'responses', 'position_ids', 'prompts', 'input_ids']
        # "inputs_ids是标识符" prompts才是用于生成的
        response_token_ids = gen_cut_batch.batch["responses"]
        # attention_mask = gen_cut_batch.batch["attention_mask"] # 恢复attention_mask引用
        inputs = data_processor.tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
        
        cut_response_idx = [int(len(item) * self.config.actor_rollout_ref.rollout.cut_keep_rate) for item in response_token_ids]
        # cut_response_idx = int(len(response_token_ids) * self.config.actor_rollout_ref.rollout.cut_keep_rate)  # 修正整数转换
        for idx in cut_response_idx:
            cut_response = response_token_ids[:, :idx]
        outputs = data_processor.tokenizer.batch_decode(cut_response, skip_special_tokens=True) 
        inputs = [p+r for p,r in zip(inputs,outputs)]
        
        

        '''
        那所以prompt和response我应该从DataProto中的batch解码获得，
        剩下的从non_tensor_Dict中获得，然后重新进行一遍处理和后处理的步骤是吗
        '''
        # 现在image,prompt,attentionmask等都需要重新处理并用类似的逻辑变回from_dict的形式
        
        # 我应该将这个batch按照RLHF的逻辑处理成一个batch，然后再用from_single_dict进行加载
        processed_row_dicts=[]
        max_prompt_length = self.config.data.max_prompt_length
        truncation = self.config.get("truncation","right") # 源代码中全是error
        need_tools_kwargs = self.config.get("need_tools_kwargs", False)
        iteration = 0
        image_patch_size = self.config.get("image_patch_size", 14)


        for i in range(len(gen_cut_batch)):#1024
            item = gen_cut_batch[i]
            row_dict:dict = {} # item.batch['input_ids'].size:torch.Size([768])
            multi_modal_data = {} # 
            original_images = item.non_tensor_batch['multi_modal_data']['image'] # non_tensor_batch的keys ['data_source', 'reward_model', 'extra_info', 'uid', 'index', 'interaction_kwargs', 'tools_kwargs', 'ability', 'multi_modal_inputs']
            noise_imgs = image_augment_from_PIL(original_images) #  
            # noise_imgs = [process_image(image, image_patch_size=image_patch_size) for image in noise_imgs] # 处理并存储在images列表中
            multi_modal_data["image"] = noise_imgs # 直接用original_images也是错了

            augment_inputs = data_processor(
                text=[inputs[i]],images=noise_imgs,videos=None,return_tensors="pt"
            )
            # todo1:按照scs的processor外面初始化封装传进来（全新的）
            # todo2:用vllm直接加载
            input_ids = augment_inputs.pop("input_ids")
            attention_mask = augment_inputs.pop("attention_mask")

            if "second_per_grid_ts" in augment_inputs:
                augment_inputs.pop("second_per_grid_ts")
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(augment_inputs)
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

            
            # 注意，我这块先用了truncation = 'right'试验一下
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_prompt_length,  # 匹配数据集max_prompt_length 这个没有引用到
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,  # 匹配数据集左填充规则
                truncation='right'  # 匹配数据集截断策略（error/left/right/middle）
            )

            position_ids = compute_position_id_with_mask(attention_mask)

            row_dict["input_ids"] = input_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["position_ids"] = position_ids[0]

            

            raw_prompt_ids = self.tokenizer.encode(inputs[i],add_special_tokens=False) # inputs格式错误
            if len(raw_prompt_ids) > max_prompt_length: # max_prompt_length 没定义，self.truncation没定义
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_length :]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
                elif truncation == "middle":
                    left_half = max_prompt_length // 2
                    right_half = max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif truncation == "error":
                    raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")
                
            row_dict["raw_prompt_ids"] = raw_prompt_ids
            if "extra_info" not in row_dict or row_dict["extra_info"] is None:
                row_dict["extra_info"] = dict()
            index = row_dict.get("extra_info", {}).get("index", 0)
            tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
            interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
            
            need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", need_tools_kwargs)
            if need_tools_kwargs and not tools_kwargs:
                logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
            row_dict["index"] = index
            row_dict["tools_kwargs"] = tools_kwargs
            row_dict["interaction_kwargs"] = interaction_kwargs
            processed_row_dicts.append(row_dict)
            iteration += 1

        # 本地实现list_of_dict_to_dict_of_list辅助逻辑（因未找到全局函数）
        
        # 将row_dict列表转换为DataProto格式
        print("\n DEBUG:Iteration is {}\n".format(iteration))
        batch_dict = default_collate_fn(processed_row_dicts)
        return DataProto.from_single_dict(batch_dict)
    def _truncate_batch(self,gen_cut_batch:DataProto):
        
        import verl.utils.torch_functional as verl_F    

        messages = gen_cut_batch.non_tensor_batch['before_message']
        '''
        
        '''
        response_token_ids = gen_cut_batch.batch['responses']
        # 那不对啊，他们其实都被填充到一样的长度了
        cut_response_idx = [int(len(item)*self.config.actor_rollout_ref.rollout.cut_keep_rate) for item in response_token_ids] 
        cut_response = [response_token_ids[i][:idx] for i,idx in enumerate(cut_response_idx)]
        outputs = self.processor.tokenizer.batch_decode(cut_response,skip_special_tokens=True)
        # raw_prompts = gen_cut_batch.non_tensor_batch['full_prompts']
        # cut_response = [p+r for p,r in zip(raw_prompts,outputs)]
        # cut_response[0]是一个str
        
        cut_response = []
        for message,output in zip(messages,outputs): # message是模板
            message_template = message.copy()
            message_template.append({
                "role":"assistant",
                "content":[{'type':'text','text':output}]
            })
            cut_response.append(self.processor.apply_chat_template(message_template,add_generation_prompt=True,tokenizer=False))
        
        # cut_response[0]:
        

        processed_row_dicts=[]
        max_prompt_length = self.config.data.max_prompt_length
        truncation = self.config.get("truncation","right") # 源代码中全是error
        need_tools_kwargs = self.config.get("need_tools_kwargs", False)
        iteration = 0
        image_patch_size = self.config.get("image_patch_size", 14)
        endoftext_token_id = int(151643)

        for i in range(len(gen_cut_batch)):
            
            item = gen_cut_batch[i]
            row_dict:dict = {}
            multi_modal_data = {}
            images = item.non_tensor_batch['multi_modal_data']['image']
            noise_imgs = image_augment_from_PIL(images) #  
            # noise_imgs = [process_image(image, image_patch_size=image_patch_size) for image in noise_imgs] # 处理并存储在images列表中
            multi_modal_data["image"] = noise_imgs
            augment_inputs = self.processor(
                text = cut_response[i], images = noise_imgs, videos = None, return_tensors='pt'
            )
            input_ids = augment_inputs.pop("input_ids")
            attention_mask = augment_inputs.pop("attention_mask")

            if "second_per_grid_ts" in augment_inputs:
                augment_inputs.pop("second_per_grid_ts")
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(augment_inputs)
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

            
            # 注意，我这块先用了truncation = 'right'试验一下
            # 这块让input_ids全都炸了
            
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_prompt_length,  # 匹配数据集max_prompt_length 这个没有引用到
                pad_token_id=self.tokenizer.pad_token_id, # self.tokenizer.pad_toke_id = 151643
                left_pad=True,  # 匹配数据集左填充规则
                truncation='right'  # 匹配数据集截断策略（error/left/right/middle）
            )


            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from verl.models.transformers.qwen3_vl import get_rope_index
                else:
                    from verl.models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=augment_inputs.get("image_grid_thw"),
                    video_grid_thw=augment_inputs.get("video_grid_thw"),
                    second_per_grid_ts=augment_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )  # (3, seq_length)
                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
            elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
                from verl.models.transformers.glm4v import get_rope_index

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=augment_inputs.get("image_grid_thw"),
                    video_grid_thw=augment_inputs.get("video_grid_thw"),
                    attention_mask=attention_mask[0],
                )  # (3, seq_length)
                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)

            
            is_endoftext = (input_ids[0]== endoftext_token_id)
            end_count = torch.sum(is_endoftext).item()
            
            
            row_dict["input_ids"] = input_ids[0]
            row_dict["attention_mask"] = attention_mask[0]
            row_dict["position_ids"] = position_ids[0]

            

            raw_prompt_ids = self.tokenizer.encode(cut_response[i],add_special_tokens=False) # inputs格式错误
            if len(raw_prompt_ids) > max_prompt_length: # max_prompt_length 没定义，self.truncation没定义
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_length :]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
                elif truncation == "middle":
                    left_half = max_prompt_length // 2
                    right_half = max_prompt_length - left_half
                    raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                elif truncation == "error":
                    raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")
                
            row_dict["raw_prompt_ids"] = raw_prompt_ids
            if "extra_info" not in row_dict or row_dict["extra_info"] is None:
                row_dict["extra_info"] = dict()
            index = row_dict.get("extra_info", {}).get("index", 0)
            tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
            interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
            
            need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", need_tools_kwargs)
            if need_tools_kwargs and not tools_kwargs:
                logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
            row_dict["index"] = index
            row_dict["tools_kwargs"] = tools_kwargs
            row_dict["interaction_kwargs"] = interaction_kwargs
            processed_row_dicts.append(row_dict)
            iteration += 1

        # 本地实现list_of_dict_to_dict_of_list辅助逻辑（因未找到全局函数）
        
        # 将row_dict列表转换为DataProto格式
        print("\n DEBUG:Iteration is {}\n".format(iteration))
        batch_dict = default_collate_fn(processed_row_dicts)
        return DataProto.from_single_dict(batch_dict)


        
            

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking( # Tracking继承用于记录训练中的各种指标
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0 # global是全局都能看到的计数器，他对学习率调度，检查点管理等有重要用途

        # load checkpoint before doing anything
        self._load_checkpoint() # 没有ckpt就直接训练，有的话就加载ckpt

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # 初始验证： 如果配置允许且存在验证奖励函数，则在训练开始前执行一次验证。
        # ---------------------------
        # -- 为了快我跳过了初始验证    ---
        # ---------------------------- # 我在后面加了个and False
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True) and False:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return # 如果val_only == True 的话，直接退出

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences() 
        # 跳过rollout步骤 包裹 Rollout： 这是一种调试或特殊模式，可以跳过实际的序列生成步骤，直接使用预先生成的或缓存的数据，以加速测试算法逻辑。

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False #根据global设置判断是否要对下一个步骤进行性能分析
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
# --------  Part1:训练循环初始化与数据准备-----
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {} # 各种性能指标
                timing_raw = {} # 计算各阶段的耗时

                # 启动性能分析： 如果 curr_step_profile 为 True，则启动配置的性能分析工具（如 PyTorch Profiler, Nsight Systems 等）。
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling( # 用内置方法启动性能分析
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                batch: DataProto = DataProto.from_single_dict(batch_dict) # 将batch_dict转换为DataProto格式，便于后续的ray分布式训练
                # raw_propmt[0]:'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Use a Pythagorean Triple to find x. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.<|im_end|>\n<|im_start|>assistant\n'
                # before_message[0]:
                '''
                [{'content': [...], 'role': 'user'}]
                content[0]有：[{'type': 'image'}, {'type': 'text', 'text': 'Use a Pythagorean Triple to find x. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.'}]
                # 文本插入位置会有影响吗
                content[1]有：[{'type': 'image'}, {'type': 'text', 'text': 'Find the perimeter of the triangle. Round to the nearest hundredth. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.'}]
                outputs[0]:'<think>\nThe right-angled triangle with legs of 14 and 48 is a 9-12-15 Pythagorean triple. We know that in a 9-12-15 right triangle, the hypotenuse (the side opposite the right angle) is 15. However, we observed the leg of 14, not 9. The correct Pythagorean triple that matches this will be scaled up similarly. Let\'s re-examine the similar triпы and the largest common factor to set our leg lengths correctly:\n<code>\n|  | <strong>Edge 1</strong> | <strong>Edge 2</strong>| <strong>Edge 3</strong>\n|---|---|---|---|\n| <strong>9</strong> | 9x | 12x | 15x |\n| <strong>12</strong> | 12x | 18x | 24x |\n| <strong>15</strong> | 15x | 20x | 25x |\n| <strong>6</strong> | 6x | 9x | 12x |\n| <strong>2</strong> | 2x | 3x | 5x |\n</code>\nOnly 2, 3, and 5 are prime and have no multiples in the given pairs. If scaled up, the smallest factor from here is likely 3 giving the right leg� 42. Another option (65) tried was suboptimal. Finally, the 3, 4, 5 triangle fits then this pair is known as the "Pythagorean Triple."\n</think>\n\\boxed{37}'
                '''
                # add uid to batch 为批次中的每一个样本生成唯一标识符 后续聚合和一致性检查
                # dict_keys(['data_source', 'ability', 'reward_model', 'extra_info', 'multi_modal_data', 'multi_modal_inputs', 'raw_prompt_ids', 'index', 'tools_kwargs', 'interaction_kwargs'])
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                
                multi_modal_data = batch.non_tensor_batch['multi_modal_data'] # 这个可能得像上面UID那样的？
                # 还是应该用prompt_id去分组，因为每个prompt_id对应着一个原始的prompt，而不是每个response_id
                
                ## ⏰⏰ TODO：understand
                gen_batch = self._get_gen_batch(batch) # 调用内部方法从batch中提取相关数据
                # 他是一个数据容器，承载了准备用于序列生成的输入数据
                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat( # 重复几次的原因是我们每次rollout可能会生成G个响应，这个多个响应的操作就是通过repeat实现的
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                is_last_step = self.global_steps >= self.total_training_steps # 判断是否当前训练为最后一轮
# --------  Part2:序列生成-----
                '''
                SCS对于一致性采样和加噪声的逻辑是
                1.对于每个原始的prompt生成rollout_n个响应 
                2.对于每个response 进行截断，并每个生成cut_num个响应， 将这些所有的轨迹合起来计算奖励（这里面要注意分组，因为cut_num对应着原始的propmt/response_id）
                3.进行更新
                '''
                with marked_timer("step", timing_raw): # 这个marked_timer是个上下文管理器，这次是把step和gen过程的耗时记录
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else: # 异步模式下用这个
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"]) # 字典信息更新
                        gen_batch_output.meta_info.pop("timing", None) # 从gen_batch_output的meta_info中弹出timing字段，更新到timing_raw中
                    ## ⏰⏰ TODO：understand（但是应该不用，我们是GRPO）
                    # REMAX 是一种特殊的优势估计方法，它通过比较 实际采样序列的奖励 与 当前策略下贪婪生成的最优序列的奖励 来计算优势。
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX: # 如果是这个的话会额外跑一次贪心生成以得到基线轨迹
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    # gen_batch_output.batch.keys():_StringKeys(dict_keys(['input_ids', 'responses', 'position_ids', 'attention_mask', 'prompts']))
                    # gen_batch_output.non_tensor_batch.keys():dict_keys(['ability', 'multi_modal_inputs', 'before_message', 'index', 'full_prompts', 'interaction_kwargs', 'tools_kwargs'])
                
                    # batch.non_tensor_batch的keys ['data_source', 'reward_model', 'extra_info', 'uid']
                    batch.non_tensor_batch['multi_modal_data'] = multi_modal_data
                    # batch.non_tensor_batch:dict_keys(['data_source', 'reward_model', 'extra_info', 'uid'])
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # rollout.n条response对应的uid是相同的
                    batch = batch.union(gen_batch_output) # batch生成完毕   
                    assert 'uid' in batch.non_tensor_batch, "uid is in the non_tensor_batch"
                    # gen_batch_output.non_tensor_batch.keys():dict_keys(['interaction_kwargs', 'index', 'ability', 'multi_modal_inputs', 'tools_kwargs']) 
# --------  Part3.1 Cut_Response 的生成
                    # 这个batch的长度是batchsize*rollout
                    # 这块缺少截断逻辑：先截断在散开，先把cut_batch中的prompt和response分别提取出来，然后截断
                    # 先截断再生成_get_gen_batch
                    cut_batch = deepcopy(batch)
                    
                    cut_batch = self._truncate_batch(cut_batch) 
                    # 这里就已经出现了endoftext
                    gen_cut_batch = self._get_gen_batch(cut_batch)
                    
                    gen_cut_batch.meta_info["global_steps"] = self.global_steps
                    gen_cut_batch_output = gen_cut_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.cut_n, interleave=True
                    ) # actor_rollout_cut_num是需要往config中添加的内容，需要更改config的内容 
                    
                    is_last_step = self.global_steps >= self.total_training_steps # 这个和上面原始rollout生成重复了，需要判断他的逻辑是否有问题
                    with marked_timer("step_cut",timing_raw):
                        with marked_timer("gen_cut",timing_raw,color="blue"):
                            if not self.async_rollout_mode:
                                gen_cut_batch_output = self.actor_rollout_wg.generate_sequences(gen_cut_batch_output)
                            else:
                                gen_cut_batch_output = self.async_rollout_manager.generate_sequences(gen_cut_batch_output)
                            timing_raw.update(gen_cut_batch_output.meta_info["timing"])
                            gen_cut_batch_output.meta_info.pop("timing", None) # 从gen_batch_output的meta_info中弹出timing字段，更新到timing_raw中
                    
                    
                    cut_batch = cut_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.cut_n, interleave=True)
                    cut_batch = cut_batch.union(gen_cut_batch_output) # 已经带有response
                    # assert 'uid' in cut_batch.non_tensor_batch, "uid is in the cut:non_tensor_batch"
                    # 目前 TODO：1.合并的逻辑还没有写 ✅ 2.如何找到cut_response对应的分组 3.rewardmanager的逻辑更改
                    # 在SCS原版的论文代码中，response和cut_response是分开存储的；所以还要了解DataProto的储存原理以及dict是如何转化成DataProto进行训练的
                    # 如果都是按照顺序存储且原始response和cut_response分开存储的话，那其实可以对于第i个原始response，就可以是[i*cut_num,(i+1)*cut_num]将对应的cut_response取出来
                    # 如果这样的话，其实在rewardm                                                                                                                                                             anager里面更改逻辑就好了
# --------  Part3:批次均衡与全局 token 计数-----
                    if "response_mask" not in batch.batch.keys(): # response_mask是标记恢复中哪些是响应的内容
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).

                    if self.config.trainer.balance_batch: # 批次平衡
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    '''
                    利用AttentionMask计算全局有效token,例如，假设我们有三个句子：
                    1. "Hello world" (2 tokens)
                    2. "How are you?" (3 tokens)
                    3. "I am fine, thank you." (5 tokens)
                    如果我们将它们组成一个批次，并填充到最长序列的长度（5 tokens），可能会变成这样：
                    1. "Hello world [PAD] [PAD] [PAD]"
                    2. "How are you? [PAD] [PAD]"
                    3. "I am fine, thank you."
                    这里的 [PAD] 就是填充的 Token。
                    attention_mask 的作用：为了让模型知道哪些是真实的 Token，哪些是填充的 Token，我们通常会使用一个 attention_mask 。
                    - attention_mask 是一个与输入序列长度相同的二进制张量。
                    - 对于真实的 Token， attention_mask 的值为 1 。
                    - 对于填充的 Token， attention_mask 的值为 0 。
                    以上面的例子为例，对应的 attention_mask 可能如下
                    1. [1, 1, 0, 0, 0] 2. [1, 1, 1, 0, 0] 3. [1, 1, 1, 1, 1]
                    '''
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
# -------- Part3.2 Cut_Response 的生成

# --------  Part4:奖励计算与模型更新-----
                    # 奖励阶段
                    with marked_timer("reward", timing_raw, color="yellow"): # 记录奖励信息
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys(): # 如果使用了奖励模型 ( self.use_rm ) 且 batch 中还没有奖励模型分数 奖励模型指的是ORM PRM那一类
                            reward_tensor = self.rm_wg.compute_rm_score(batch) # 对这个批次进行打分
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async: # 异步计算
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn) # 将各个来源的奖励组合
                            # 其中reward_ten是tensor类型，extra是字典类型，和奖励有关的额外的信息

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    # 旧策略和熵指标？
# --------  Part5:旧策略对数概率计算与熵指标-----
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    # PPO的思想就是通过限制新策略和就策略之间的差异防止策略更新过快，导致训练不稳定

                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction
                        '''
                        使用 rollout_log_probs : 在这种模式下，代码的注释明确指出 # Use rollout_log_probs``。
                        这意味着 old_log_probs 不会通过 self.actor_rollout_wg.compute_log_prob(batch) 重新计算。
                        相反，它会直接使用在数据收集（rollout）阶段就已经计算并存储在 batch 中的 rollout_log_probs 。
                        他的意思就直接在前向传播的过程中顺道给算了，就不用再重新调用一次forward计算了
                        '''
                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            # PPO的标准做法就是利用当前的actor模型计算所有动作的对数概率
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch) # 计算旧策略的对数概率
                            # 熵可能会用到损失函数中
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss( # 将单个时间步/token的损失熵组合成一个单一的用于梯度计算的损失值，不同的聚合方式会影响到损失更新的方向
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics) # 更新
                            old_log_prob.batch.pop("entropys") # 熵已经被更新到metric了，不需要存储了
                            batch = batch.union(old_log_prob) # 将新计算出来的old_log_prob更新到batch中保证后续可以收到并使用数据
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}' # 保证old_log_prob防止后续计算出错
# --------  Part6:参考策略与Critic计算-----
                    # 什么是用reference_policy：防止我们的策略和原有策略不要相差太大
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:# 如果不集成在actor中，这可能需要加载和维护两个独立的模型（Actor 和 Reference），从而 增加内存占用和计算资源 。但它提供了更大的灵活性，例如参考策略可以是完全不同的架构，或者可以独立于 Actor 进行更新（例如，保持冻结）。
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else: # 这通常通过在 Actor 模型中添加一个额外的输出头或共享大部分底层权重来实现。这种方式可以 节省内存和计算资源 ，因为只需要加载和运行一个模型。
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch) # 计算value
                            batch = batch.union(values)
# --------  Part7:优势计算与策略更新-----
                    # 奖励的最后处理阶段
                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor # 之前由compute_reward_fn计算得到的奖励张量

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward: # 是否应用KL散度
                            # ⏰⏰
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )  
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        '''
                        我们希望能够知道序列中 每个决策（即生成每个 Token）的好坏 。如果每个 Token 的奖励都相同，那么
                        - 它本质上就退化成了序列级别的奖励，只是被广播到了每个 Token。
                        - 策略很难区分序列中哪些 Token 的生成是好的，哪些是差的，从而难以进行有效的学习和改进。
                        '''

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if ( # 旁路模式决定了到底需不需要重新计算一遍old_log_probs
                            rollout_corr_config is not None # Rollout 修正是在旁路模式下，为了 弥补直接使用 rollout_log_probs 可能带来的不准确性 而引入的机制。即使我们直接使用了数据收集阶段的 rollout_log_probs ，这些值可能与当前训练策略所需的“旧策略”对数概率存在差异
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True # 获取GRPO归一化的一些参数
                        )  # GRPO adv normalization factor

                        # 使用自己的配置计算优势
                        # ⏰⏰ 计算优势
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            # 更新critic
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
## ----- 用计算完的优势和reward更新actor和critic
                    # implement critic warmup 训练开始的时候先只更新critic模型而保持actor模型不变（得保证价值估计要稳定一点）
                    if self.config.trainer.critic_warmup <= self.global_steps: # 如果评论家的预热步数已经达到了，才能更新actor
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"): # 先让critic稳定下来再更新actor
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # 更新actor
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"]) # 抽取Actor模型的指标并用reduce_metrics进行聚合
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        # 记录第一轮rollout
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                        cut_rollout_dir = '/data1/yyy25/verl/cut_rollout'
                        self._cut_log_rollout_data(batch=cut_batch,rollout_data_dir=cut_rollout_dir)


                # validate #
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
# --------  Part8:检查点保存-----
                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                    '''
                    - 这通常是指云服务提供商（如 AWS EC2 Spot Instances、Google Cloud Preemptible VMs 或阿里云抢占式实例）提供的 可中断的、成本较低的计算实例 。这些实例的价格通常远低于按需实例，但云提供商保留随时回收这些实例的权利，通常会提前几分钟（例如，AWS Spot Instance 会提前 2 分钟）发出通知。
            - close_to_expiration ：当云提供商发出实例即将被回收的通知时，这个状态就会变为 True 。
            为什么在训练脚本中会关注这个状态？ 在机器学习训练，特别是长时间运行的训练任务中，为了降低成本，研究人员和工程师经常会使用这些抢占式实例。然而，实例被回收意味着训练任务可能会中断，导致已完成的工作丢失。
'''
# --------  Part9:性能分析停止与计时更新-----
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

# part10 指标收集记录
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
