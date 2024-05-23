from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


from .multimodal_components.vision_tower_builder import build_vision_tower
from .multimodal_components.mm_projector_builder import build_vision_projector
from .multimodal_components.mm_align import AlignIndicator, GatedWeightLayer

from src.utils.constants import *

class AlignGPTConfig(LlamaConfig):
    model_type = 'aligngpt'

class AlignGPTForCausalLM(LlamaForCausalLM):
    config_class = AlignGPTConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if hasattr(config, "mm_vision_tower"): # In inference stage
            # initialize align components
            self.align_indicators = AlignIndicator(config)
            self.gated_weight_layer = GatedWeightLayer(config)

            # initialize vision model & mm projector
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
            
    def set_stage(self, stage):
        assert stage in STAGES
        self.stage = stage
     
    def initialize_align_components(self):
        self.align_indicators = AlignIndicator(self.config)
        if self.stage == FINETUNE or self.stage == INFERENCE:
            self.gated_weight_layer = GatedWeightLayer(self.config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model() # load weights of clip model

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
    
    def load_pretrained_weights(self, model_args):
        if self.stage == FINETUNE or self.stage == INFERENCE:
            pretrain_mm_mlp_align = model_args.pretrain_mm_mlp_align
            if pretrain_mm_mlp_align is not None:
                mlp_align_weights = torch.load(pretrain_mm_mlp_align, map_location='cpu')
            
                def get_w(weights, keyword):
                    w_dict = {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                    return w_dict

                def get_w_2(weights, keyword):
                    w_dict = {k.split('.')[-1]: v for k, v in weights.items() if keyword in k}
                    return w_dict
                
                self.mm_projector.load_state_dict(get_w(mlp_align_weights, 'mm_projector'))
                self.align_indicators.load_state_dict(get_w_2(mlp_align_weights, 'indicator_embs'))
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        align_ids: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                past_key_values,
                conc_var_group,
                text_var_group,
                image_var_group,
                pure_text_var_group
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
            
            position_ids, attention_mask, inputs_embeds, labels = conc_var_group


            if self.stage == 'pretrain':    
                align_indicators = self.align_indicators(align_ids)
                align_indicators = align_indicators.unsqueeze(dim=1)
                
                inputs_embeds = torch.cat([inputs_embeds[:, :1, :], align_indicators, inputs_embeds[:, 1:, :]], dim=1)
                
            elif self.stage == 'finetune' or self.stage == 'inference':
                if images is not None and input_ids is None:
                    text_position_ids, text_attention_mask, text_embeds, text_labels = text_var_group
                    image_position_ids, image_attention_mask, image_embeds, image_labels = image_var_group
                    pure_text_position_ids, pure_text_attention_mask, pure_text_embeds, pure_text_labels = pure_text_var_group
                    
                    if len(image_embeds) > 0:
                        w_scores = self.gated_weight_layer.forward(image_embeds, text_embeds[:, 1:, :], text_attention_mask[:, 1:])
                        align_indicators = self.align_indicators.lin_comb(w_scores).unsqueeze(dim=1)
                        if self.stage == 'inference': align_indicators = align_indicators.to(device=inputs_embeds.device)
                        
                        inputs_embeds = torch.cat([inputs_embeds[:, :1, :], align_indicators, inputs_embeds[:, 1:, :]], dim=1)
                    
                    if pure_text_embeds is not None:
                        if len(image_embeds) == 0:
                            dummy_scores = self.gated_weight_layer.dummy_forward(len(pure_text_embeds), pure_text_embeds.device, pure_text_embeds.dtype)
                            dummy_al_id = self.align_indicators.lin_comb(dummy_scores).unsqueeze(dim=1)
                            pure_text_embeds = torch.cat([pure_text_embeds, dummy_al_id[:, 0:0, :]], dim=1)
                        # in_pure = True
                        inputs_embeds = torch.cat([inputs_embeds, pure_text_embeds]) if len(inputs_embeds) > 0 else torch.cat([pure_text_embeds])
                        labels = torch.cat([labels, pure_text_labels]) if labels is not None else pure_text_labels
                        attention_mask = torch.cat([attention_mask, pure_text_attention_mask]) if attention_mask is not None else pure_text_attention_mask
                        position_ids = torch.cat([position_ids, pure_text_position_ids]) if position_ids is not None else pure_text_position_ids
                               
                    if len(image_embeds) == 0 and pure_text_embeds is None:
                        raise ValueError(f'Shouldn\'t reach here')
                
            else:
                raise ValueError(f'Unsupported stage name: {self.stage}')
                     
        else:
            raise ValueError(f'Shouldn\'t reach here')
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    def encode_images(self, images):
        image_features = self.get_vision_tower()(images)
        image_features = self.mm_projector(image_features)
        return image_features
    
    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values,
        labels, 
        images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # no images
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                # for generation
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            
            return input_ids, past_key_values, \
                (position_ids, attention_mask, None, labels), \
                None, \
                None, \
                None

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images).to(self.device)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist, it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea, please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        image_embeds, image_labels = [], []
        text_embeds, text_labels = [], []
        
        new_input_embeds, new_labels = [], []
        pure_text_embeds, pure_text_labels = None, None

        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                
                if pure_text_embeds is None and pure_text_labels is None:
                    pure_text_embeds, pure_text_labels = [], []
                
                pure_text_embeds.append(cur_input_embeds)
                pure_text_labels.append(labels[batch_idx])
                
                cur_image_idx += 1
            
                continue
            
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            
            cur_new_input_embeds, cur_new_labels = [], []
            cur_text_embeds, cur_text_labels = [], []
            cur_image_embeds, cur_image_labels = [], []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                
                cur_text_embeds.append(cur_input_embeds_no_im[i])
                cur_text_labels.append(cur_labels_noim[i])
                
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    pad_im_labels = torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                    
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(pad_im_labels)

                    cur_image_embeds.append(cur_image_features)
                    cur_image_labels.append(pad_im_labels)
            
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
                    
            cur_text_embeds = torch.cat(cur_text_embeds)
            cur_text_labels = torch.cat(cur_text_labels)
            text_embeds.append(cur_text_embeds)
            text_labels.append(cur_text_labels)
            
            cur_image_embeds = torch.cat(cur_image_embeds)
            cur_image_labels = torch.cat(cur_image_labels)
            image_embeds.append(cur_image_embeds)
            image_labels.append(cur_image_labels)
                    
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        
        if len(new_input_embeds) > 0:
            if tokenizer_model_max_length is not None:
                real_model_max_length = tokenizer_model_max_length - N_INDICATOR_TOKEN
                new_input_embeds = [x[:real_model_max_length] for x in new_input_embeds]
                new_labels = [x[:real_model_max_length] for x in new_labels]
            
            # Combine them
            assert len(text_embeds) == len(text_labels)
            assert len(text_embeds) == len(new_input_embeds)
            
            max_len = max(x.shape[0] for x in new_input_embeds)
            max_text_len = max(x.shape[0] for x in text_embeds)
            max_image_len = max(x.shape[0] for x in image_embeds)
            
            im_batch_size = len(new_input_embeds)

            new_input_embeds_padded = []
            new_labels_padded = torch.full((im_batch_size, max_len+N_INDICATOR_TOKEN), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            attention_mask = torch.zeros((im_batch_size, max_len+N_INDICATOR_TOKEN), dtype=attention_mask.dtype, device=attention_mask.device)
            position_ids = torch.zeros((im_batch_size, max_len+N_INDICATOR_TOKEN), dtype=position_ids.dtype, device=position_ids.device)
            
            text_embeds_padded = []
            text_labels_padded = torch.full((im_batch_size, max_text_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            text_attention_mask = torch.zeros((im_batch_size, max_text_len), dtype=attention_mask.dtype, device=attention_mask.device)
            text_position_ids = torch.zeros((im_batch_size, max_text_len), dtype=position_ids.dtype, device=position_ids.device)

            image_embeds_padded = []
            image_labels_padded = torch.full((im_batch_size, max_image_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            image_attention_mask = torch.zeros((im_batch_size, max_image_len), dtype=attention_mask.dtype, device=attention_mask.device)
            image_position_ids = torch.zeros((im_batch_size, max_image_len), dtype=position_ids.dtype, device=position_ids.device)

            for i, (new_embed_cur, new_labels_cur, 
                    text_embed_cur, text_labels_cur,
                    image_embed_cur, image_labels_cur) in enumerate(zip(new_input_embeds, new_labels,
                                                                        text_embeds, text_labels,
                                                                        image_embeds, image_labels)):
                cur_len = new_embed_cur.shape[0]
                cur_text_len = text_embed_cur.shape[0]
                cur_image_len = image_embed_cur.shape[0]
                
                if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                    new_input_embeds_padded.append(torch.cat((
                        torch.zeros((max_len - cur_len, new_embed_cur.shape[1]), dtype=new_embed_cur.dtype, device=new_embed_cur.device),
                        new_embed_cur
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -(cur_len+N_INDICATOR_TOKEN):-(cur_len+N_INDICATOR_TOKEN-1)] = new_labels_cur[:1]
                        new_labels_padded[i, -(cur_len+N_INDICATOR_TOKEN-1):-(cur_len-1)] = torch.full((N_INDICATOR_TOKEN,), IGNORE_INDEX, device=new_labels_cur.device, dtype=new_labels_cur.dtype)
                        new_labels_padded[i, -(cur_len-1):] = new_labels_cur[1:]
                        attention_mask[i, -(cur_len+N_INDICATOR_TOKEN):] = True
                        position_ids[i, -(cur_len+N_INDICATOR_TOKEN):] = torch.arange(0, cur_len+N_INDICATOR_TOKEN, dtype=position_ids.dtype, device=position_ids.device)
                    
                    text_embeds_padded.append(torch.cat((
                        torch.zeros((max_text_len - cur_text_len, text_embed_cur.shape[1]), dtype=text_embed_cur.dtype, device=text_embed_cur.device),
                        text_embed_cur
                    ), dim=0))
                    if cur_text_len > 0:
                        text_labels_padded[i, -cur_text_len:] = text_labels_cur
                        text_attention_mask[i, -cur_text_len:] = True
                        text_position_ids[i, -cur_text_len:] = torch.arange(0, cur_text_len, dtype=position_ids.dtype, device=position_ids.device)
                    
                    image_embeds_padded.append(torch.cat((
                        torch.zeros((max_image_len - cur_image_len, image_embed_cur.shape[1]), dtype=image_embed_cur.dtype, device=image_embed_cur.device),
                        image_embed_cur
                    ), dim=0))
                    if cur_image_len > 0:
                        image_labels_padded[i, -cur_image_len:] = image_labels_cur
                        image_attention_mask[i, -cur_image_len:] = True
                        image_position_ids[i, -cur_image_len:] = torch.arange(0, cur_image_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    new_input_embeds_padded.append(torch.cat((
                        new_embed_cur,
                        torch.zeros((max_len - cur_len, new_embed_cur.shape[1]), dtype=new_embed_cur.dtype, device=new_embed_cur.device)
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :1] = new_labels_cur[:1]
                        new_labels_padded[i, 1:N_INDICATOR_TOKEN+1] = torch.full((N_INDICATOR_TOKEN,), IGNORE_INDEX, device=new_labels_cur.device, dtype=new_labels_cur.dtype)
                        new_labels_padded[i, N_INDICATOR_TOKEN+1:N_INDICATOR_TOKEN+cur_len] = new_labels_cur[1:]
                        attention_mask[i, :cur_len+N_INDICATOR_TOKEN] = True
                        position_ids[i, :cur_len+N_INDICATOR_TOKEN] = torch.arange(0, cur_len+N_INDICATOR_TOKEN, dtype=position_ids.dtype, device=position_ids.device)
                    
                    text_embeds_padded.append(torch.cat((
                        text_embed_cur,
                        torch.zeros((max_text_len - cur_text_len, text_embed_cur.shape[1]), dtype=text_embed_cur.dtype, device=text_embed_cur.device)
                    ), dim=0))
                    if cur_text_len > 0:
                        text_labels_padded[i, :cur_text_len] = text_labels_cur
                        text_attention_mask[i, :cur_text_len] = True
                        text_position_ids[i, :cur_text_len] = torch.arange(0, cur_text_len, dtype=position_ids.dtype, device=position_ids.device)
                    
                    image_embeds_padded.append(torch.cat((
                        image_embed_cur,
                        torch.zeros((max_image_len - cur_image_len, image_embed_cur.shape[1]), dtype=image_embed_cur.dtype, device=image_embed_cur.device)
                    ), dim=0))
                    if cur_image_len > 0:
                        image_labels_padded[i, :cur_image_len] = image_labels_cur
                        image_attention_mask[i, :cur_image_len] = True
                        image_position_ids[i, :cur_image_len] = torch.arange(0, cur_image_len, dtype=position_ids.dtype, device=position_ids.device)
            
            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
            text_embeds = torch.stack(text_embeds_padded, dim=0)
            image_embeds = torch.stack(image_embeds_padded, dim=0)
        
        if pure_text_embeds is not None:
            if len(new_input_embeds) > 0:
                max_pure_len = new_input_embeds.shape[1] + N_INDICATOR_TOKEN
            else:
                max_pure_len = max(x.shape[0] for x in pure_text_embeds)
            pure_text_embeds = [x[:max_pure_len] for x in pure_text_embeds]
            pure_text_labels = [x[:max_pure_len] for x in pure_text_labels]
            
            pure_batch_size = len(pure_text_embeds)
            
            pure_text_embeds_padded = []
            pure_text_labels_padded = torch.full((pure_batch_size, max_pure_len), IGNORE_INDEX, dtype=labels[0].dtype, device=labels[0].device)
            pure_text_attention_mask = torch.zeros((pure_batch_size, max_pure_len), dtype=attention_mask.dtype, device=attention_mask.device)
            pure_text_position_ids = torch.zeros((pure_batch_size, max_pure_len), dtype=position_ids.dtype, device=position_ids.device)
            
            for i, (cur_pure_embed, cur_pure_labels) in enumerate(zip(pure_text_embeds, pure_text_labels)):
                cur_pure_len = cur_pure_embed.shape[0]
                if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                    pure_text_embeds_padded.append(torch.cat((
                        torch.zeros((max_pure_len - cur_pure_len, cur_pure_embed.shape[1]), dtype=cur_pure_embed.dtype, device=cur_pure_embed.device),
                        cur_pure_embed
                    ), dim=0))
                    if cur_pure_len > 0:
                        pure_text_labels_padded[i, -cur_pure_len:] = cur_pure_labels
                        pure_text_attention_mask[i, -cur_pure_len:] = True
                        pure_text_position_ids[i, -cur_pure_len:] = torch.arange(0, cur_pure_len, dtype=position_ids.dtype, device=position_ids.device)
                else:
                    pure_text_embeds_padded.append(torch.cat((
                        cur_pure_embed,
                        torch.zeros((max_pure_len - cur_pure_len, cur_pure_embed.shape[1]), dtype=cur_pure_embed.dtype, device=cur_pure_embed.device)
                    ), dim=0))
                    if cur_pure_len > 0:
                        pure_text_labels_padded[i, :cur_pure_len] = cur_pure_labels
                        pure_text_attention_mask[i, :cur_pure_len] = True
                        pure_text_position_ids[i, :cur_pure_len] = torch.arange(0, cur_pure_len, dtype=position_ids.dtype, device=position_ids.device)
            
            pure_text_embeds = torch.stack(pure_text_embeds_padded, dim=0)
        
        if len(new_input_embeds) == 0 and pure_text_embeds is None:
            raise ValueError(f'Shouldn\'t reach here')
        

        if _labels is None:
            new_labels = None
            text_labels = None
            image_labels = None
            pure_text_labels = None
        else:
            new_labels = new_labels_padded if len(new_input_embeds) > 0 else None
            text_labels = text_labels_padded if len(new_input_embeds) > 0 else None
            image_labels = image_labels_padded if len(new_input_embeds) > 0 else None
            pure_text_labels = pure_text_labels_padded if pure_text_embeds is not None else None

        if _attention_mask is None:
            attention_mask = None
            text_attention_mask = None
            image_attention_mask = None
            pure_text_attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype) if len(new_input_embeds) > 0 else None
            text_attention_mask = text_attention_mask.to(dtype=_attention_mask.dtype) if len(new_input_embeds) > 0 else None
            image_attention_mask = image_attention_mask.to(dtype=_attention_mask.dtype) if len(new_input_embeds) > 0 else None
            pure_text_attention_mask = pure_text_attention_mask.to(dtype=_attention_mask.dtype) if pure_text_embeds is not None else None 
            

        if _position_ids is None:
            position_ids = None
            text_position_ids = None
            image_position_ids = None
            pure_text_position_ids = None
        else:
            position_ids = position_ids if len(new_input_embeds) > 0 else None
            text_position_ids = text_position_ids if len(new_input_embeds) > 0 else None
            image_position_ids = image_position_ids if len(new_input_embeds) > 0 else None
            pure_text_position_ids = pure_text_position_ids if pure_text_embeds is not None else None
            
    
        return None, past_key_values, \
            (position_ids, attention_mask, new_input_embeds, new_labels), \
            (text_position_ids, text_attention_mask, text_embeds, text_labels), \
            (image_position_ids, image_attention_mask, image_embeds, image_labels), \
            (pure_text_position_ids, pure_text_attention_mask, pure_text_embeds, pure_text_labels)
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


AutoConfig.register("aligngpt", AlignGPTConfig)
AutoModelForCausalLM.register(AlignGPTConfig, AlignGPTForCausalLM)
