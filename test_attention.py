from verl.utils.transformers_compat import is_transformers_version_in_range
print("import complete \n")

if is_transformers_version_in_range(min_version="4.54.0"):
            print("First Branch \n")
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
elif is_transformers_version_in_range(min_version="4.53.0"):
            print("Second Branch \n")
            raise RuntimeError("Transformers 4.53.* is bugged. Use transformers 4.54.0 or later.")
else:
            print("Third branch \n")
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLFlashAttention2 as Qwen2_5_VLAttention,
            )
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2 as Qwen2VLAttention