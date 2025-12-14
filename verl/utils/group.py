from verl import DataProto
from mathruler.grader import extract_boxed_content

def group_id_dict(cut_batch:DataProto, processor=None) -> dict:
    '''
    要将cut_batch中的uid和response分组，uid是要重新分配的，response是要计算奖励的
    重构后结构：{'uid': {'cut_sign': [对应答案列表]}}
    '''
    group_id_dict = {}
    for i in range(len(cut_batch)):
        data_item = cut_batch[i]
        uid = data_item.non_tensor_batch["uid"]
        # 获取non_tensor_batch中的cut_sign作为子字典的key
        cut_sign = data_item.non_tensor_batch["cut_sign"]
        # 1. 解码 token ID 得到原始字符串
        decoded_response = processor.tokenizer.decode(
            data_item.batch["responses"], 
            skip_special_tokens=True
        )
        
        # 2. 提取 boxed content
        extracted = extract_boxed_content(decoded_response)
        
        # 3. 使用 object() 作为 None 的唯一占位符
        if extracted == 'None':
            processed_response = object() # 每个 None 都会得到一个新的 object 实例
        else:
            processed_response = extracted # 这是一个答案字符串
            
        # 初始化uid对应的子字典
        if uid not in group_id_dict:
            group_id_dict[uid] = {}
        # 初始化cut_sign对应的答案列表
        if cut_sign not in group_id_dict[uid]:
            group_id_dict[uid][cut_sign] = []
        # 将处理后的答案加入对应cut_sign的列表
        group_id_dict[uid][cut_sign].append(processed_response)
 
    return group_id_dict