import torch
from typing import Dict
from transformers import AutoModel
from accelerate import init_empty_weights, infer_auto_device_map

def split_model(pretrained_model_name_or_path: str, max_memory: Dict[str, any]):
    """Tính toán và chia nhỏ các layer của mô hình AI lên nhiều GPU/CPU để tránh tràn RAM (OOM)."""
    with init_empty_weights():
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        dtype=torch.bfloat16
    )
    return device_map

def disable_flash_attn(module, target_attrs=['use_flash_attn']):
    """Đệ quy quét và tắt tính năng Flash Attention để mô hình chạy an toàn trên mọi loại phần cứng."""
    for child_name, child in module.named_children():
        disable_flash_attn(child, target_attrs)

    for attr in target_attrs:
        if hasattr(module, attr):
            if getattr(module, attr) is True:
                setattr(module, attr, False)