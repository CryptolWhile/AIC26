import torch
import torch.nn as nn
from typing import Optional
from transformers import BlipForImageTextRetrieval


class BlipModel(BlipForImageTextRetrieval):

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs
    ) -> torch.FloatTensor:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding
        )

        image_embeds = vision_outputs[0]
        image_feat = nn.functional.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> torch.FloatTensor:

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        text_feat = nn.functional.normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
        return text_feat