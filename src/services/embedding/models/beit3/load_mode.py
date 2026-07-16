from typing import Any, Tuple

import torch
from transformers import XLMRobertaTokenizer

from src.services.embedding.models.beit3.modeling_finetune import (
    beit3_base_patch16_224_retrieval,
    beit3_large_patch16_384_retrieval
)
from src.services.embedding.models.beit3.utils import load_model_and_may_interpolate


def load_beit3_model(
    model_name: str = "beit3_base_patch16_224_retrieval"
) -> Tuple[Any, Any]:
    
    tokenizer = XLMRobertaTokenizer("./src/services/embedding/models/beit3/weights/beit3.spm")
    
    if model_name == "beit3_base_patch16_224_retrieval":
        checkpoint_path = "./src/services/embedding/models/beit3/weights/beit3_base_itc_patch16_224.pth"
        model_checkpoint = torch.load(checkpoint_path)
        model = beit3_base_patch16_224_retrieval(pretrained=True)
        model.load_state_dict(model_checkpoint['model'])
    
    elif model_name == "beit3_large_patch16_384_retrieval":
        checkpoint_path = "./src/services/embedding/models/beit3/weights/beit3_large_itc_patch16_224.pth"
        model = beit3_large_patch16_384_retrieval(pretrained=True)
        load_model_and_may_interpolate(checkpoint_path, model, model_key='model', model_prefix='')
        
    return tokenizer, model