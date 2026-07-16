import torch
import torch.nn as nn
from typing import Optional
from transformers import Blip2ForImageTextRetrieval


class Blip2Model(Blip2ForImageTextRetrieval):

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
            return_dict=return_dict
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )
        image_embeds = query_outputs[0] if not return_dict else query_outputs.last_hidden_state

        image_embeds = nn.functional.normalize(self.vision_projection(image_embeds), dim=-1)
        return image_embeds

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> torch.FloatTensor:

        query_embeds = self.embeddings(input_ids=input_ids)
        text_outputs = self.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = text_outputs[0] if not return_dict else text_outputs.last_hidden_state
        text_embeds = nn.functional.normalize(self.text_projection(question_embeds[:, 0, :]), dim=-1)
        return text_embeds