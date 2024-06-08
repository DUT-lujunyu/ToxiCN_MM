import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel, ViTModel
from transformers import ChineseCLIPModel


class MHKE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cv_path = config.vit_path
        self.nlp_path = config.chinese_roberta_path
        self.cv_model = ViTModel.from_pretrained(self.cv_path)
        self.nlp_model = BertModel.from_pretrained(self.nlp_path)
        # self.dropout = nn.Dropout(p=0.1)
        self.attention = QKVAttention()
        self.classifier = nn.Linear(config.hidden_dim*2, config.num_classes)
        self.device = config.device
        self.weight = config.weight

    def forward(self, **args):
        text_outputs = self.nlp_model(input_ids=args['input_ids'].to(self.device),
                                      attention_mask=args['attention_mask'].to(self.device))
        text_discription_outputs = self.nlp_model(input_ids=args['text_discription_input_ids'].to(self.device),
                                                  attention_mask=args['text_discription_attention_mask'].to(self.device))
        meme_discription_outputs = self.nlp_model(input_ids=args['meme_discription_input_ids'].to(self.device),
                                                  attention_mask=args['meme_discription_attention_mask'].to(self.device))

        # text_features = self.dropout(text_outputs['pooler_output'])
        # text_discription_features = self.dropout(
        #     text_discription_outputs['pooler_output'])
        # meme_discription_features = self.dropout(
        #     meme_discription_outputs['pooler_output'])

        # text_with_k_s = self.attention(text_outputs['pooler_output'],
        #                                text_discription_outputs['pooler_output'], text_outputs['pooler_output'])[0]
        # text_with_k_v = self.attention(text_outputs['pooler_output'],
        #                                meme_discription_outputs['pooler_output'], text_outputs['pooler_output'])[0]

        # text_with_k = torch.mean(torch.stack(
        #     [text_outputs['pooler_output'], text_with_k_s]), dim=0)
        text_with_k = text_outputs['pooler_output'] + \
            self.weight * meme_discription_outputs['pooler_output'] + text_discription_outputs['pooler_output']

        # text_with_k = text_outputs['pooler_output'] + \
        #     text_discription_outputs['pooler_output']
        # text_with_k = text_outputs['pooler_output'] + \
        #     meme_discription_outputs['pooler_output']
        # text_with_k = text_outputs['pooler_output'] + \
        #     text_discription_outputs['pooler_output'] + \
        #     meme_discription_outputs['pooler_output']

        image_outputs = self.cv_model(
            pixel_values=args['image_tensor'].to(self.device))

        features = torch.cat(
            (text_with_k, image_outputs['pooler_output']), dim=1)

        output = self.classifier(features)
        return output


class MHKE_CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.path = config.chinese_clip_path
        self.model = ChineseCLIPModel.from_pretrained(self.path)
        self.attention = QKVAttention()
        self.classifier = nn.Linear(config.hidden_dim*2, config.num_classes)
        self.device = config.device

    def forward(self, **args):
        text_features = self.model.get_text_features(input_ids=args['input_ids'].to(self.device),
                                                     attention_mask=args['attention_mask'].to(self.device))
        text_discription_features = self.model.get_text_features(input_ids=args['text_discription_input_ids'].to(self.device),
                                                                 attention_mask=args['text_discription_attention_mask'].to(self.device))
        meme_discription_features = self.model.get_text_features(input_ids=args['meme_discription_input_ids'].to(self.device),
                                                                 attention_mask=args['meme_discription_attention_mask'].to(self.device))

        # text_with_k_v = self.attention(text_features,
        #                                meme_discription_features, text_features)[0]

        # text_with_k = text_features + text_discription_features
        # text_with_k = text_features + meme_discription_features
        text_with_k = text_features + text_discription_features + meme_discription_features

        image_features = self.model.get_image_features(
            pixel_values=args['image_tensor'].to(self.device))

        features = torch.cat((text_with_k, image_features), dim=1)
        output = self.classifier(features)
        return output


class QKVAttention(nn.Module):
    def __init__(self):
        super(QKVAttention, self).__init__()

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        matmul_qk = torch.matmul(Q, K.transpose(-2, -1))
        dk = K.shape[-1]
        scaled_attention_logits = matmul_qk / \
            torch.sqrt(torch.tensor(dk).float())

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, Q, K, V, mask=None):
        output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask)
        return output, attention_weights
