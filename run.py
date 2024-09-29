import os
import numpy as np

import torch
import torch.nn as nn
import json

from torchvision.transforms import functional as F
# from transformers import BertTokenizer, CLIPProcessor, CLIPModel, CLIPTokenizer

from config.Config_base import Config_base
from dataset.dataset import *
from train_eval_ import train, test

from model.clip import *
from model.vit_roberta import *
from model.MHKE import *

if __name__ == '__main__':

    model_name = "clip"
    # model_name = "vit-roberta"
    task_name = "task_2"
    config = Config_base(model_name, task_name)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    if not os.path.exists(config.data_path): 
        trn_data = MemeDataset(config, training=True)
        test_data = MemeDataset(config, training=False)
        torch.save({
            'trn_data' : trn_data,
            'test_data' : test_data,
            }, config.data_path)
    else:
        checkpoint = torch.load(config.data_path)
        trn_data = checkpoint['trn_data']
        test_data = checkpoint['test_data']

    print('The size of the Training dataset: {}'.format(len(trn_data)))
    print('The size of the Test dataset: {}'.format(len(test_data)))

    train_iter = DataLoader(trn_data, batch_size=int(config.batch_size), shuffle=False)
    test_iter = DataLoader(test_data, batch_size=int(config.batch_size), shuffle=False)

    train(config, train_iter, test_iter)

    # weights = [0.5]
    # for weight in weights:
    #     config.weight = weight
    #     train(config, train_iter, test_iter)

    # all_batch_size = [16, 32, 64]
    # for batch_size in all_batch_size:
    #     config.batch_size = batch_size
    #     train(config, train_iter, test_iter)

    # learning_rate = [1e-4, 5e-5]
    # for batch_size in learning_rate:
    #     config.learning_rate = batch_size
    #     train(config, train_iter, test_iter)

    # model = MHKE(config).to(config.device)
    # path = '{}/ckp-MHKE_B-32_E-10_Lr-1e-05_w-0.5_task_1_add-BEST.tar'.format(config.checkpoint_path, model_name, 'BEST')
    # checkpoint = torch.load(path)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # preds = test(model, test_iter)



