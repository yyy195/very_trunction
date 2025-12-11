from verl import DataProto
from mathruler.grader import extract_boxed_content

def group_id_dict(cut_batch:DataProto, processor=None) -> dict:
    '''
    要将cut_batch中的uid和response分组，uid是要重新分配的，response是要计算奖励的
    '''
    group_id_dict = {}
    for i in range(len(cut_batch)):
        data_item = cut_batch[i]
        tag_uid = data_item.non_tensor_batch["tag_uid"]
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
            
        if tag_uid not in group_id_dict:
            group_id_dict[tag_uid] = [processed_response]
        else:
            group_id_dict[tag_uid].append(processed_response)
 
    return group_id_dict