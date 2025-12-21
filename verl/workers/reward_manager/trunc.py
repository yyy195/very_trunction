from collections import defaultdict,Counter
from typing import Any

import torch
import math
import re

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from mathruler.grader import extract_boxed_content, grade_answer

def max_infoentro_increase(answer_dict:dict) -> tuple[float, Any, dict, dict]:
    max_entropy = 0.0
    max_entropy_cut_sign = None
    all_cut_sign_counts = Counter()
    max_entropy_specific_counts = Counter() # 新增：存储最大熵对应的cut_sign的答案计数

    for cut_sign, answer_list in answer_dict.items():
        counts = Counter(answer_list)
        all_cut_sign_counts.update(counts)

        total_elements = len(answer_list)
        if total_elements == 0:
            continue

        entropy = 0.0
        for count in counts.values():
            probability = count / total_elements
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        if entropy > max_entropy:
            max_entropy = entropy
            max_entropy_cut_sign = cut_sign
            max_entropy_specific_counts = counts # 更新为当前cut_sign的计数
            
    return max_entropy, max_entropy_cut_sign, dict(all_cut_sign_counts), dict(max_entropy_specific_counts)

def calculate_adjusted_reward(
    current_trajectory_answer: Any,
    uid_set:list,
    total_all_answers: int, # 新增参数
    boost_factor: float = 0.2, # 提升因子
    penalty_factor: float = 0.1 # 惩罚因子
) -> float:
    """计算调整后的奖励。"""

    count = uid_set.count(current_trajectory_answer)
    # 1. 计算基础频率奖励
    if total_all_answers == 0:
        base_frequency = 0.0
    else:
        base_frequency = count / total_all_answers
    
    base_reward = base_frequency # 基础奖励可以简单地等于这个频率

    # 2. 根据max_cut_sign进行调整
    final_reward = base_reward
    total_max_cut_sign_answers = len(uid_set)
    if total_max_cut_sign_answers > 0:
        max_cut_sign_frequency = count / total_max_cut_sign_answers
    else:
        max_cut_sign_frequency = 0.0

    # 调整逻辑
    if max_cut_sign_frequency == 0 and base_frequency > 0:
        # 情况1: 答案在整体中存在，但在max_cut_sign中不存在。
        # 这可能表明这是一个“难以找到”的答案，因此我们提升奖励。
        final_reward += base_reward * boost_factor
    elif max_cut_sign_frequency > base_frequency and base_frequency > 0:
        # 情况2: 答案在max_cut_sign中的频率高于整体频率。提升奖励。
        final_reward += base_reward * boost_factor
    elif max_cut_sign_frequency < base_frequency and max_cut_sign_frequency > 0:
        # 情况3: 答案在max_cut_sign中的频率低于整体频率。降低奖励。
        final_reward -= base_reward * penalty_factor
    
    # 确保奖励在合理范围 [0, 1]
    return max(0.0, min(1.0, final_reward))


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

def consistency_reward(predict_str: str, cut_answer_set:list) -> float:
    
    all_answers_for_diversity = []
    
    # 1. 处理预测答案 (predict_str)
    extracted_from_predict = extract_boxed_content(predict_str)
    
    # 使用 object() 作为 None 的占位符
    if extracted_from_predict is None:
        # 如果预测的答案也是 None，也用一个新的 object() 来占位
        all_answers_for_diversity.append(object()) 
    else:
        all_answers_for_diversity.append(extracted_from_predict)

    # 2. 加入 cut_answer_set 的所有元素
    # cut_answer_set 应该已经包含字符串和 object() 实例
    all_answers_for_diversity.extend(cut_answer_set) 

    # 3. 计算多样性 (Diversity)
    # set() 会自动将相同的字符串答案合并，并保持所有独特的 object() 实例分开
    diversity = len(set(all_answers_for_diversity))
    
    return (5 - diversity) * 0.25
    

def compute_score_majority(predict_str: str, ground_truth: str, cut_answer_set:list, use_boxed: bool = True, format_score: float = 0.1) -> float:
    # 注意有所更改，这个cut_answer_set是list，每个list都存着cut_num个response，extract_boxed_answer完了
    return (0.9 - 2 * format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(predict_str) + (format_score + 0.1) * consistency_reward(predict_str,cut_answer_set) 






@register("trunc")
class TruncRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", rollout_n=2, flag=True, cut_num=3, val_rollout_n=1) -> None:
        
        '''
        ToDo1:传进来的数据形式最简单粗暴的就是batch:DataProto 和 cut_batch:DataProto
        ToDo2:如何确定标识符，原文给予的uid对于rollout_n是唯一的，但是对于cut_batch来说，uid是重复的，所以可能需要以相同的方法再生成uid
        ToDo3:需要写分类函数，将prompt和截断的response分在一组计算奖励
        -------
        1.其实我们只需要cut_batch的uid和responses就可以了，uid用于分组，responses用于计算奖励
        2.cut_batch在外面整理好再传输进来，要整理成一个TensorDict的形式{'uid':'response'}，uid是要重新分配的，response
        '''
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.rollout_n = rollout_n
        self.cut_num = cut_num
        self.flag = flag
        self.val_rollout_n = val_rollout_n

    def __call__(self, data: DataProto, cut_batch:dict,  return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
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

        uid_list = []
        per_pro_rollout_n = self.rollout_n
        answers_box = []
        valid_length_box = []
        response_content = []
        

        assert per_pro_rollout_n > 1, "If you use MajorityVotes, Rollout_n should be greater than 1"

        for i in range(len(data)): # 处理原始batch

            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            uid = data_item.non_tensor_batch['tag_uid']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.flag == False: # 评估的时候要用groundtruth
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
                uid_list.append(uid)
        
        # 处理cut_batch

        # 以下是Majority投票议程

        if self.flag == True:
            # 训练阶段
            # 我知道了！如果所有的都没有提取出答案，那么answer_box都会是None，然后多数投票投出来的就是None，如果再提取出来的也是None，这个score就是1了
            
            original_batch = len(data) // per_pro_rollout_n
            majority_answer = []
            for i in range(original_batch): # 用于算单纯的答案的
                group_answers = answers_box[i * per_pro_rollout_n : (i + 1) * per_pro_rollout_n]
                group_counter = Counter(group_answers)
                group_majority = group_counter.most_common(1)[0][0] # most_common(1)只会返回最多的元素和对应的次数[('A',5)]
                majority_answer.extend([group_majority] * per_pro_rollout_n)

            for i in range(original_batch):
                uid_rollout = uid[i * per_pro_rollout_n : (i + 1) * per_pro_rollout_n]
                uid_set = []
                for _,value in uid_rollout.items():
                    uid_set.extend(value)

                # _, _, all_cut_sign_counts_dict, max_cut_sign_specific_counts_dict = max_infoentro_increase(cut_batch_statistic)         
                
                total_all_answers = len(uid_set)
                for j in range(per_pro_rollout_n):
                    current_trajectory_index = i * per_pro_rollout_n + j
                    current_trajectory_answer = answers_box[current_trajectory_index]
                    valid_response_length = valid_length_box[current_trajectory_index]

                    freq_score = calculate_adjusted_reward(
                        current_trajectory_answer,
                        uid_set,
                        total_all_answers,
                    ) 
                    format_score = format_reward(response_content[current_trajectory_index])
                    score = 0.9 * freq_score + 0.1 * format_score
                    reward_tensor[current_trajectory_index, valid_response_length-1] = score
            '''
            for i in range(len(data)):
                tag_uid = data[i].non_tensor_batch["tag_uid"] # 当前轨迹的uid
                valid_response_length = valid_length_box[i]
                if answers_box[i] == 'None': # 如果他不为None，那么Majority也必定有不为None的，所以就不为None
                    score = 0.0
                else: # 此时可以传入答案集合，因为其他的rollout都是辅助的 cut_batch[uid]是truction的答案辅助集合
                    score = compute_score_majority(response_content[i],majority_answer[i],cut_batch[tag_uid]) # 这个计算的是原始output字符串，但是MM-UPT他是选择题
                reward_tensor[i, valid_response_length-1] = score
            '''
        # 到这里应该出现reward_tensor和reward_extra_info
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor



'''
改成MMUPT+频率reward+固定ratio截断
以下是distribution_trunc的代码
from collections import defaultdict,Counter
from typing import Any

import torch
import re
import math
from collections import Counter

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from mathruler.grader import extract_boxed_content, grade_answer


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

def consistency_reward(predict_str: str, cut_answer_set:list) -> float:
    
    all_answers_for_diversity = []
    
    # 1. 处理预测答案 (predict_str)
    extracted_from_predict = extract_boxed_content(predict_str)
    
    # 使用 object() 作为 None 的占位符
    if extracted_from_predict is None:
        # 如果预测的答案也是 None，也用一个新的 object() 来占位
        all_answers_for_diversity.append(object()) 
    else:
        all_answers_for_diversity.append(extracted_from_predict)

    # 2. 加入 cut_answer_set 的所有元素
    # cut_answer_set 应该已经包含字符串和 object() 实例
    all_answers_for_diversity.extend(cut_answer_set) 

    # 3. 计算多样性 (Diversity)
    # set() 会自动将相同的字符串答案合并，并保持所有独特的 object() 实例分开
    diversity = len(set(all_answers_for_diversity))
    
    return (5 - diversity) * 0.25
    

def compute_score_majority(predict_str: str, ground_truth: str, cut_answer_set:list, use_boxed: bool = True, format_score: float = 0.1) -> float:
    # 注意有所更改，这个cut_answer_set是list，每个list都存着cut_num个response，extract_boxed_answer完了
    return (0.9 - 2 * format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(predict_str) + (format_score + 0.1) * consistency_reward(predict_str,cut_answer_set) 



def max_infoentro_increase(answer_dict:dict) -> tuple[float, Any, dict, dict]:
    max_entropy = 0.0
    max_entropy_cut_sign = None
    all_cut_sign_counts = Counter()
    max_entropy_specific_counts = Counter() # 新增：存储最大熵对应的cut_sign的答案计数

    for cut_sign, answer_list in answer_dict.items():
        counts = Counter(answer_list)
        all_cut_sign_counts.update(counts)

        total_elements = len(answer_list)
        if total_elements == 0:
            continue

        entropy = 0.0
        for count in counts.values():
            probability = count / total_elements
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        if entropy > max_entropy:
            max_entropy = entropy
            max_entropy_cut_sign = cut_sign
            max_entropy_specific_counts = counts # 更新为当前cut_sign的计数
            
    return max_entropy, max_entropy_cut_sign, dict(all_cut_sign_counts), dict(max_entropy_specific_counts)






@register("trunc")
class TruncRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", rollout_n=2, flag=True, cut_num=3, val_rollout_n=1) -> None:
        
        
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.rollout_n = rollout_n
        self.cut_num = cut_num
        self.flag = flag
        self.val_rollout_n = val_rollout_n

    def __call__(self, data: DataProto, cut_batch:dict,  return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
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

        uid_list = []
        per_pro_rollout_n = self.rollout_n
        answers_box = []
        valid_length_box = []
        response_content = []
        

        assert per_pro_rollout_n > 1, "If you use MajorityVotes, Rollout_n should be greater than 1"

        for i in range(len(data)): # 处理原始batch

            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            uid = data_item.non_tensor_batch["uid"]
            

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.flag == False: # 评估的时候要用groundtruth
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
                uid_list.append(uid) # 加入到uid列表中
        
        # 处理cut_batch

        # 以下是Majority投票议程
        if self.flag == True:
            # 训练阶段
            # 我知道了！如果所有的都没有提取出答案，那么answer_box都会是None，然后多数投票投出来的就是None，如果再提取出来的也是None，这个score就是1了
            
            original_batch = len(data) // per_pro_rollout_n
            majority_answer = []
           
            for i in range(original_batch): 
                # {'uid': {'cut_sign': [对应答案列表]}}
                group_answers = answers_box[i * per_pro_rollout_n : (i + 1) * per_pro_rollout_n] # group_answers是uid相同的rollout.n条的answer
                valid_response_group = valid_length_box[i * per_pro_rollout_n : (i + 1) * per_pro_rollout_n]
                uid_rollout = uid_list[i * per_pro_rollout_n : (i + 1) * per_pro_rollout_n]
                uid = uid_rollout[0]    
                for item in uid_rollout: # 保证这一组里面的uid都是相等的
                    assert item == uid, f"uid({uid}) is not equal to item({item})"
                # cut_batch是这样的结构：{'uid': {'cut_sign': [对应答案列表]}}
                cut_batch_statistic = cut_batch[uid] # cut_batch_statistics是对应uid的子字典 
                # 本身就算没有截断
                # 接下来要计算最大信息增益，统计辅助集合中不同答案的数量
                info_entropy, max_cut_sign, all_cut_sign_counts_dict, max_cut_sign_specific_counts_dict = max_infoentro_increase(cut_batch_statistic)         
                
                total_all_answers = sum(all_cut_sign_counts_dict.values())

                # 为当前uid对应的所有轨迹计算奖励
                for j in range(per_pro_rollout_n):
                    current_trajectory_index = i * per_pro_rollout_n + j
                    current_trajectory_answer = answers_box[current_trajectory_index]
                    valid_response_length = valid_length_box[current_trajectory_index]

                    # 还没有算格式奖励那些
                    freq_score = calculate_adjusted_reward(
                        current_trajectory_answer,
                        all_cut_sign_counts_dict,
                        max_cut_sign_specific_counts_dict,
                        total_all_answers
                    ) 
                    format_score = format_reward(response_content[current_trajectory_index])
                    score = 0.9 * freq_score + 0.1 * format_score
                    reward_tensor[current_trajectory_index, valid_response_length-1] = score
            
        # 到这里应该出现reward_tensor和reward_extra_info
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor




'''