import torch
import torch.nn as nn
from transformers import ChineseCLIPModel


class CLIPMemesClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.path = config.clip_path
        self.model = ChineseCLIPModel.from_pretrained(self.path)
        self.classifier = nn.Linear(config.hidden_dim*2, config.num_classes)
        self.device = config.device
  
    def forward(self, **args):
        text_features = self.model.get_text_features(input_ids = args['input_ids'].to(self.device),
                                                    attention_mask = args['attention_mask'].to(self.device))
        image_features = self.model.get_image_features(pixel_values = args['image_tensor'].to(self.device))

        features = torch.cat((text_features, image_features), dim=1)
        output = self.classifier(features)
        return output