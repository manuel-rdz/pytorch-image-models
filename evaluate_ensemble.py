import argparse
import time
from timm.utils.metrics import mAP_score
from torch._C import dtype
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import numpy as np

from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint, load_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
# from timm.data import LoadImagesAndLabels,preprocess,LoadImagesAndLabelsV2,LoadImagesAndSoftLabels
from timm.utils import ApexScaler, auc_score
from timm.utils import Visualizer
from timm.data import get_riadd_train_transforms, get_riadd_valid_transforms,get_riadd_test_transforms
from timm.data import RiaddDataSet,RiaddDataSet9Classes

import os
from tqdm import tqdm
import random
import torch.distributed as dist
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

CFG = {
    'seed': 42,
    'img_size': 224,
    'valid_bs': 10,
    'num_workers': 4,
    'num_classes': 29,
    'tta': 3,
    'models': [#'b6-ns-768/tf_efficientnet_b6_ns-768-fold0-model_best.pth.tar',
               #'b5-ns-960/tf_efficientnet_b5_ns-960-fold0-model_best.pth.tar',
               '20210910-205105-vit_base_patch16_384-384/model_best.pth.tar'],
    'base_img_path': 'C:/Users/AI/Desktop/student_Manuel/datasets/RIADD_cropped/Evaluation_Set/Evaluation',
    'weights': [1]
}



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def validate(model, loader): 
    model.eval()
    preds = []
    pbar = tqdm(enumerate(loader), total=len(loader))  
    with torch.no_grad():
        for batch_idx, (input, target) in pbar:
            input = input.cuda()
            target = target.cuda()
            target = target.float()
            output = model(input)
            preds.append(output.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

if __name__ == '__main__':
    from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
    import pandas as pd
    import torch.utils.data as data
    seed_everything(CFG['seed'])
    data_ = pd.read_csv('C:/Users/AI/Desktop/student_Manuel/datasets/RIADD_cropped/Evaluation_Set/RFMiD_Validation_Labels.csv')
    test_index = [i for i in range(data_.shape[0])]

    test_data = data_.iloc[test_index, :].reset_index(drop=True)
    #print(test_data.head())
    #print(test_index) 
    test_transforms = get_riadd_test_transforms(CFG)
    test_dataset = RiaddDataSet(image_ids = test_data,transform = test_transforms, baseImgPath = CFG['base_img_path'])
    test_data_loader = data.DataLoader( test_dataset, 
                                        batch_size=CFG['valid_bs'], 
                                        shuffle=False, 
                                        num_workers=CFG['num_workers'], 
                                        pin_memory=True, 
                                        drop_last=False,
                                        sampler = None)
    
    imgIds = test_data.iloc[test_index,0].tolist()
    target_cols = test_data.iloc[test_index, 1:].columns.tolist()
    #print(target_cols)    
    test = pd.DataFrame()
    test['ID'] = imgIds

    tst_preds = []
    for i,model_name in enumerate(CFG['models']):
        model_path = os.path.join('C:/Users/AI/Desktop/student_Manuel/codes/RIADD_1st_place/pretrained/',model_name)
        model = create_model(model_name = 'tf_efficientnet_b6_ns',num_classes=CFG['num_classes'])

        if model_name.find('ResNet200D') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'resnet200d',num_classes=CFG['num_classes'])
        
        if model_name.find('nf_resnet50') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'nf_resnet50',num_classes=CFG['num_classes'])
        
        if model_name.find('tf_efficientnet_b7_ns') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'tf_efficientnet_b7_ns',num_classes=CFG['num_classes'])
        
        if model_name.find('tf_efficientnet_b4_ns') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'tf_efficientnet_b4_ns',num_classes=CFG['num_classes'])

        if model_name.find('tf_efficientnet_b6_ns') != -1:
            os.path.join('C:/Users/AI/Desktop/student_Manuel/codes/RIADD_1st_place/pretrained/', model_name)
            model = create_model(model_name = 'tf_efficientnet_b6_ns',num_classes=CFG['num_classes'])

        if model_name.find('tf_efficientnet_b5_ns') != -1:
            os.path.join('C:/Users/AI/Desktop/student_Manuel/codes/RIADD_1st_place/pretrained/', model_name)
            model = create_model(model_name = 'tf_efficientnet_b5_ns',num_classes=CFG['num_classes'])

        if model_name.find('vit_base_patch16_224') != -1:
            model_path = os.path.join('C:/Users/AI/Desktop/student_Manuel/codes/pytorch_image_models/pytorch-image-models/output/train/', model_name)
            model = create_model(model_name='vit_base_patch16_224', num_classes=CFG['num_classes'])

        if model_name.find('vit_base_patch16_384') != -1:
            model_path = os.path.join('C:/Users/AI/Desktop/student_Manuel/codes/pytorch_image_models/pytorch-image-models/output/train/', model_name)
            model = create_model(model_name='vit_base_patch16_384', num_classes=CFG['num_classes'])
        
        print('model_path: ',model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict["state_dict"], strict=True)
        model = nn.DataParallel(model)
        model.cuda()
        for _ in range(CFG['tta']):
            tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*validate(model,test_data_loader)]
    tst_preds = np.sum(tst_preds, axis=0)
    test_ = test_data.iloc[test_index, 1:].to_numpy(dtype=np.int32)

    '''auc, scores_auc = auc_score(test_[:, 1:], tst_preds)
    mAP, scores_mAP = mAP_score(test_[:, 1:], tst_preds)
    task2_score = (auc + mAP) / 2

    print('----- Multilabel scores -----')
    print('auc_score: ', auc)
    print('mAP: ', mAP)
    print('task score: ', task2_score)

    test[target_cols[1:]] = tst_preds'''
 
    auc_bin, scores_auc = auc_score(test_[:, 0], tst_preds[:, 0])
    
    auc, scores_auc = auc_score(test_[:, 1:], tst_preds[:, 1:])
    mAP, scores_mAP = mAP_score(test_[:, 1:], tst_preds[:, 1:])
    task2_score = (auc + mAP) / 2

    final_score = (auc_bin + task2_score) / 2
    print('----- Multilabel scores -----')
    print('auc_score: ', auc)
    print('mAP: ', mAP)
    print('task score: ', task2_score)
    print('----- Binary scores -----')
    print('auc: ', auc_bin)
    print('----- Final Score -----')
    print(final_score)
    test[target_cols] = tst_preds

    test.to_csv('submission_eb6-768-29classes.csv', index=False)