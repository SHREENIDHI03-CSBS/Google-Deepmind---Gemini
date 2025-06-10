import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import models, transforms

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V2")
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

class GeminiModel(nn.Module):
    def __init__(self, image_emb_dim=512, text_model_name="gpt2"):
        super().__init__()
        self.image_encoder = ImageEncoder(output_dim=image_emb_dim)
        self.language_model = GPT2LMHeadModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(image_emb_dim, self.language_model.config.n_embd)
    
    def forward(self, input_ids, attention_mask, images):
        img_embeds = self.image_encoder(images)
        img_embeds = self.text_proj(img_embeds).unsqueeze(1)  # Shape: (B, 1, D)
        
        inputs_embeds = self.language_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([img_embeds, inputs_embeds], dim=1)
        
        attention_mask = torch.cat([torch.ones((input_ids.size(0), 1), device=input_ids.device), attention_mask], dim=1)
        
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs
