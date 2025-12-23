from collections import defaultdict,Counter
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from mathruler.grader import extract_boxed_content, grade_answer
import re

def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    #
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score_majority(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )


@register("majority")
class MajorityRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, 
                 tokenizer, 
                 num_examine, 
                 compute_score=None, 
                 reward_fn_key="data_source", 
                 rollout_n=4, 
                 flag=True,
                 val_rollout_n=1) -> None:
        
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.rollout_n = rollout_n
        self.flag = flag # True是train False是val
        self.val_rollout_n = val_rollout_n

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        per_pro_rollout_n = self.rollout_n
        answers_box = []
        valid_length_box = []
        response_content = []
        

        assert per_pro_rollout_n > 1, "If you use MajorityVotes, Rollout_n should be greater than 1"

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.flag == False:
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
                extra_info["num_turns"] = num_turns
                extra_info["rollout_reward_scores"] = rollout_reward_scores
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                reward_tensor[i, valid_response_length-1] = score

            else:
                response_in_box = extract_boxed_content(response_str)
                response_content.append(response_str)
                answers_box.append(response_in_box)
                valid_length_box.append(valid_response_length)
            

        # 以下是Majority投票议程

        if self.flag == True:
            # 训练阶段
            # 将奖励设置为当前答案在rollout_n条响应中的出现频率
            original_batch = len(data) // per_pro_rollout_n
            answer_frequencies = []  # 存储每个响应对应的答案出现频率
            for i in range(original_batch):
                group_start = i * per_pro_rollout_n
                group_end = (i + 1) * per_pro_rollout_n
                group_answers = answers_box[group_start:group_end]
                group_counter = Counter(group_answers)
                group_size = len(group_answers)
                # 计算组内每个答案的频率并填充到answer_frequencies
                group_frequencies = [group_counter[ans] / group_size for ans in group_answers]
                answer_frequencies.extend(group_frequencies)

            for i in range(len(data)):
                valid_response_length = valid_length_box[i]
                current_answer = answers_box[i]
                if current_answer == 'None':
                    score = 0.0
                else:
                    # 获取当前答案在组内的频率作为核心奖励
                    freq_score = answer_frequencies[i]
                    # 保留格式奖励作为补充（可根据需求调整权重比例）
                    format_score_val = format_reward(response_content[i])
                    score = 0.9 * freq_score + 0.1 * format_score_val
                reward_tensor[i, valid_response_length-1] = score

        # 到这里应该出现reward_tensor和reward_extra_info
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

'''
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
'''