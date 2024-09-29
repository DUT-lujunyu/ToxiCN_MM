import pandas as pd
import torch
import random
import time
from datetime import timedelta
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ChineseCLIPProcessor, AutoImageProcessor, BertTokenizer, ViTFeatureExtractor


class MemeDataset(Dataset):

    def __init__(self, config, training=True):
        self.config = config
        self.model_name = config.model_name
        self.tokenizer = BertTokenizer.from_pretrained(config.roberta_path)
        if self.model_name == "clip":
            self.processor = ChineseCLIPProcessor.from_pretrained(config.clip_path)
        elif self.model_name == "resnet":
            self.processor = AutoImageProcessor.from_pretrained(config.resnet_path)
        else:
            self.extractor = ViTFeatureExtractor.from_pretrained(config.vit_path)
        
        if training:
            self.data = pd.read_json(config.train_path)
        else:
            self.data = pd.read_json(config.test_path)
        self.max_len = config.pad_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row.text
        label = data_row.new_label
        type_label = data_row.new_type
        modal = data_row.modal
        file_name = data_row.new_path

        text_discription = data_row.text_discription
        meme_discription = data_row.meme_discription

        label = torch.tensor(label).float()
        type_label = torch.tensor(type_label).float()
        modal = torch.tensor(modal)
        image = Image.open(self.config.meme_path + file_name).convert('RGB')

        if self.model_name == "clip":
            text_inputs = self.processor(
                text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
            text_discription_inputs = self.processor(
                text_discription, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
            meme_discription_inputs = self.processor(
                meme_discription, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        else:
            text_inputs = self.tokenizer(
                text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
            text_discription_inputs = self.tokenizer(
                text_discription, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
            meme_discription_inputs = self.tokenizer(
                meme_discription, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")         

        if self.model_name == "clip" or self.model_name == "resnet" :
            image_inputs = self.processor(images=image, return_tensors='pt')
        else:
            image_inputs = self.extractor(image, return_tensors='pt')

        input_ids = text_inputs["input_ids"].squeeze()
        attention_mask = text_inputs["attention_mask"].squeeze()
        text_discription_input_ids = text_discription_inputs["input_ids"].squeeze()
        text_discription_attention_mask = text_discription_inputs["attention_mask"].squeeze()
        meme_discription_input_ids = meme_discription_inputs["input_ids"].squeeze()
        meme_discription_attention_mask = meme_discription_inputs["attention_mask"].squeeze()

        image_tensor = image_inputs["pixel_values"].squeeze()

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_discription_input_ids=text_discription_input_ids,
            text_discription_attention_mask=text_discription_attention_mask,
            meme_discription_input_ids=meme_discription_input_ids,
            meme_discription_attention_mask=meme_discription_attention_mask,
            image_tensor=image_tensor,
            label=label,
            type_label=type_label,
            modal=modal
        )


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def convert_onehot(config, label):
    onehot_label = [0 for i in range(config.num_classes)]
    onehot_label[int(label)] = 1
    return onehot_label
