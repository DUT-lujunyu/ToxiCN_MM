import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from os import path

class Config_base(object):

    """配置参数"""
    def __init__(self, model_name, task_name):

        # path
        self.model_name = model_name
        self.task_name = task_name
        self.clip_path = "/home/home_ex/lujunyu/models/chinese-clip-vit-base-patch16"
        self.roberta_path = "/home/home_ex/lujunyu/models/chinese-roberta-wwm-ext"
        self.bert_path = "/home/home_ex/lujunyu/models/bert-base-chinese"
        self.vit_path = "/home/home_ex/lujunyu/models/vit-base-patch16-224" 
        self.resnet_path = "/home/home_ex/lujunyu/models/resnet-50" 


        if self.model_name == "clip":               
            self.hidden_dim = 512
        else:
            self.hidden_dim = 768 

        self.meme_path = path.dirname(path.dirname(__file__)) + '/meme/'          
        self.train_path = path.dirname(path.dirname(__file__)) + '/train_data_discription.json'                                # 训练集
        self.dev_path = path.dirname(path.dirname(__file__)) + '/test_data_discription.json'                                    # 验证集
        self.test_path = path.dirname(path.dirname(__file__))+'/test_data_discription.json'  
        self.result_path = path.dirname(path.dirname(__file__))+'/result'                                # 测试集
        self.checkpoint_path = path.dirname(path.dirname(__file__))+'/saved_dict'        # 数据集、模型训练结果
        self.data_path = self.checkpoint_path + '/' + self.model_name + '_data.tar'

        if self.task_name == "task_1":
            self.num_classes = 2                                             # 类别数
        else:
            self.num_classes = 5                                             # 类别数

        # dataset
        self.seed = 1        
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)

        # model
        self.dropout = 0.5                                              # 随机失活
        self.fc_hidden_dim = 256
        self.weight = 0.5

        # train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.learning_rate = 1e-5                                       # 学习率  transformer:5e-4 
        self.num_epochs = 10                                            # epoch数 
        self.num_warm = 0                                              # 预热
        self.batch_size = 32                                           # mini-batch大小

        # evaluate
        self.score_key = "F1"                                            # 评价指标

# if __name__ == '__main__':
#     config = Config_base("BERT")
#     print(config.train_path)
