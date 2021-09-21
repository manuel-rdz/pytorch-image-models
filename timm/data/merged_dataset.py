from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import torch
import tarfile
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from PIL import Image
import cv2
import numpy as np
#from timm.data.riadd_augment import crop_maskImg

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

class MergedDataset(data.Dataset):  # for training/testing
    def __init__(self, image_ids, baseImgPath = [], selftrans = False, load_bytes=False,transform=None,onlydisease = False):
        self.image_ids = image_ids
        self.baseImgPath = baseImgPath
        self.transform = transform
        self.load_bytes = load_bytes
        self.selftrans = selftrans
        self.onlydisease = onlydisease
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        imgId = self.image_ids.iloc[index, 0]
        if self.image_ids.iloc[index, 1] == 1:
            dataset_idx = 0
        else:
            imgId = str(imgId) + '.ppm'
            dataset_idx = 1

        if self.onlydisease == True:
            label_2 = self.image_ids.iloc[index, 2:].values.astype(np.int64)
            label_2 = sum(label_2)
            label = self.image_ids.iloc[index, 1:2].values.astype(np.int64)
            if label_2 > 0:
                label = np.append(label, 1)
            else:
                label = np.append(label,0)
        else:
            label = self.image_ids.iloc[index, 3:].values.astype(np.int64)
        imgpath = os.path.join(self.baseImgPath[dataset_idx], imgId)
        if self.selftrans == True:
            img = open(path, 'rb').read() if self.load_bytes else Image.open(imgpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            #print('Tying to open image: ', imgpath)
            img = cv2.imread(imgpath)
            #img = crop_maskImg(img)
            try:
                img = img[:, :, ::-1]
            except:
                print(imgpath)
            img = self.transform(image = img)['image']
        return img, label
