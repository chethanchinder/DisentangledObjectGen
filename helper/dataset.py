from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import sys
from utils import *
import scipy.io as sio
import time
import glob
import json
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class ShapeNet(data.Dataset):
    def __init__(self,
                 objectfile="/home/chethanc/reconstruction/data/mini_shapenet_mat",
                 class_choice = "chair",
                 train = True, npoints = 2500, normal = False,
                 idx=0, extension = 'png'):
        self.normal = normal
        self.train = train
        self.objectfile = objectfile
        self.npoints = npoints
        self.datapath = []
        self.catfile = os.path.join('./extras/synsetoffset2category.txt')
        self.cat = {}
        self.meta = {}
        self.idx=idx
        self.extension = extension
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        print(self.cat)
        for object_id,item in enumerate(self.cat):
            try:
                dir_point = os.path.join(self.objectfile, self.cat[item])
                fns_pc = sorted(os.listdir(dir_point))
                
            except:
                continue

            fns = [val.split('.')[0] for val in fns_pc]
            print('category ', self.cat[item], 'files ' + str(len(fns)), len(fns)/float(len(fns_pc)), "%")
                        # total values
            fns_values = np.array(fns)
            # integer encode
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(fns_values)

            # binary encode
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            # first example
            # integer_fns = label_encoder.transform([fns_values[2]])
            # one_hot_vector = onehot_encoder.transform([integer_fns])

            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]

            
            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    shape_onehot_int = label_encoder.transform([fn])
                    shape_onehot_vec = onehot_encoder.transform([shape_onehot_int]).squeeze()
                    class_vec = np.array([object_id])
                    pose_vec =  np.array([0, 0, 0])
                    self.meta[item].append((class_vec, shape_onehot_vec,
                                    pose_vec,os.path.join(dir_point, fn+'.mat'),
                                    item, fn ) )
            
        self.idx2cat = {}
        self.size = {}
        i = 0
        for item in self.cat:
            self.idx2cat[i] = item
            self.size[i] = len(self.meta[item])
            i = i + 1
            for fn in self.meta[item]:
                self.datapath.append(fn)

        self.transforms = transforms.Compose([
                             transforms.Resize(size =  224, interpolation = 2),
                             transforms.ToTensor(),
                        ])

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()

    def __getitem__(self, index):
        fn = self.datapath[index]
        class_vec = fn[0]
        shape_onehot_vec = fn[1]
        pose_vec = fn[2]
        fp = sio.loadmat(fn[3])
        points = fp['v']
        indices = np.random.randint(points.shape[0], size=self.npoints)
        points = points[indices,:]
        if self.normal:
            normals = fp['f']
            normals = normals[indices,:]
        else:
            normals = 0
        cat = fn[4]
        name = fn[5]
        
        return class_vec, shape_onehot_vec, pose_vec, points,normals, name, cat

    def __len__(self):
        return len(self.datapath)


if __name__  == '__main__':
    print('Testing Shapenet dataset')
    dataset  =  ShapeNet(class_choice = None,
                 train = False, npoints = 10000,
                normal = True, idx=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=int(12))
    time1 = time.time()
    for i, data in enumerate(dataloader, 0):
        class_vec, shape_onehot_vec, pose_vec,points, normals, name, cat = data
        print("input shapes ", class_vec.shape, shape_onehot_vec.shape, pose_vec.shape)
        print("points shape ",points.shape,normals.shape)
        #print(cat[0],name[0],points.max(),points.min())
    time2 = time.time()
    print(time2-time1)