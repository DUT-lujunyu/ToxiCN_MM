import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, ViTModel


class VitRobertaMemesClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cv_path = config.vit_path
        self.nlp_path = config.roberta_path
        self.cv_model = ViTModel.from_pretrained(self.cv_path)
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        self.classifier = nn.Linear(config.hidden_dim*2, config.num_classes)
        self.device = config.device
  
    def forward(self, **args):
        text_outputs = self.nlp_model(input_ids = args['input_ids'].to(self.device),
                                    attention_mask = args['attention_mask'].to(self.device))
        image_outputs = self.cv_model(pixel_values = args['image_tensor'].to(self.device))

        features = torch.cat((text_outputs['pooler_output'], image_outputs['pooler_output']), dim=1)
        # print(features.shape)
        output = self.classifier(features)
        return output
    
    
class VitClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cv_path = config.vit_path
        self.cv_model = ViTModel.from_pretrained(self.cv_path)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.device = config.device
  
    def forward(self, **args):
        image_outputs = self.cv_model(pixel_values = args['image_tensor'].to(self.device))
        output = self.classifier(image_outputs['pooler_output'])
        return output
    

class ResNetClassifier(nn.Module):
    def __init__(self, config):
        super(ResNetClassifier, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, config.num_classes)
        self.device = config.device

    def forward(self, **args):
        return self.resnet50(args['image_tensor'].to(self.device))


class RobertaClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nlp_path = config.roberta_path
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.device = config.device
  
    def forward(self, **args):
        text_outputs = self.nlp_model(input_ids = args['input_ids'].to(self.device),
                                    attention_mask = args['attention_mask'].to(self.device))
        output = self.classifier(text_outputs['pooler_output'])
        return output
    

class BertClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nlp_path = config.bert_path
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        self.device = config.device
  
    def forward(self, **args):
        text_outputs = self.nlp_model(input_ids = args['input_ids'].to(self.device),
                                    attention_mask = args['attention_mask'].to(self.device))
        output = self.classifier(text_outputs['pooler_output'])
        return output