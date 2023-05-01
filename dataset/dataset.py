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

class ShapeNet(data.Dataset):
    def __init__(self,
                 objectfile="/home/chethanc/reconstruction/data/TMNetMesh/customShapeNet_mat",
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
        for item in self.cat:
            try:
                dir_point = os.path.join(self.objectfile, self.cat[item])
                fns_pc = sorted(os.listdir(dir_point))
                
            except:
                continue

            fns = [val for val in fns_pc]
            print('category ', self.cat[item], 'files ' + str(len(fns)), len(fns)/float(len(fns_pc)), "%"),
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]

            if len(fns) != 0:
                self.meta[item] = []
                for fn in fns:
                    self.meta[item].append((os.path.join(dir_point, fn), item, fn ) )
            
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
        fp = sio.loadmat(fn[0])
        points = fp['v']
        indices = np.random.randint(points.shape[0], size=self.npoints)
        points = points[indices,:]
        if self.normal:
            normals = fp['f']
            normals = normals[indices,:]
        else:
            normals = 0
        cat = fn[1]
        name = fn[2]

        return points,normals, name, cat

    def __len__(self):
        return len(self.datapath)


if __name__  == '__main__':
    print('Testing Shapenet dataset')
    dataset  =  ShapeNet(class_choice = None,
                 train = True, npoints = 10000,
                normal = True, idx=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True, num_workers=int(12))
    time1 = time.time()
    for i, data in enumerate(dataloader, 0):
        points, normals, name, cat = data
        print(points.shape,normals.shape)
        print(cat[0],name[0],points.max(),points.min())
    time2 = time.time()
    print(time2-time1)