import torch
import torch.nn as nn
from src.utils.constants import N_IMAGE_TOKEN

class AlignIndicator(nn.Module):
    def __init__(self, config, n_indicators=8) -> None:
        super().__init__()
        self.indicator_embs = nn.Parameter(nn.init.xavier_uniform_(
                torch.empty(n_indicators, 
                            config.hidden_size))
        )
    
    def __call__(self, ids) -> torch.Any:
        return self.indicator_embs[ids]
    
    def lin_comb(self, b_weight_scores):
        indicators = self.indicator_embs[:-1]
        final_indicators = torch.matmul(b_weight_scores, indicators) + self.indicator_embs[-1]

        return final_indicators
        
class GatedWeightLayer(nn.Module):
    def __init__(self, config, n_components=7) -> None:
        super().__init__()
        
        self.dim = config.hidden_size
        self.mlp = nn.Sequential(*[nn.Linear(N_IMAGE_TOKEN, 256),
                                   nn.GELU(),
                                   nn.Linear(256, n_components)])

    def forward(self, image_embeds, text_embeds, text_attention_mask):
        scale = torch.sum(text_attention_mask.int(), dim=-1, keepdim=True).to(dtype=text_embeds.dtype)
        avg_text_embeds = torch.sum(text_embeds, dim=1) / (scale + 1e-8)

        dots = torch.sum(image_embeds * avg_text_embeds.unsqueeze(dim=1), dim=-1)   

        scores = self.mlp(dots)
        scores = scores.softmax(dim=-1)

        return scores
    
    def dummy_forward(self, bs, device, dtype):
        embs1 = torch.ones([bs, N_IMAGE_TOKEN], dtype=dtype, device=device)
        out = self.mlp(embs1)

        return out


