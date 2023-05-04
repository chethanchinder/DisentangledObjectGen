from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import resnet
import numpy as np


class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = (torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


class DeformNet(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(DeformNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class Refiner(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(Refiner, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 2, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class Estimator(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(Estimator, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 1, 1)

        self.sig = nn.Sigmoid()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.sig(self.conv4(x))
        return x


class SVR_TMNet(nn.Module):
    def __init__(self,  bottleneck_size = 1024,class_dim =1, shape_dim = 5000 , pose_dim = 3):
        super(SVR_TMNet, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.class_dim = class_dim
        self.shape_dim = shape_dim
        self.pose_dim = pose_dim
        self.encoder = EmbeddingNet(class_dim = self.class_dim, 
                        shape_dim = self.shape_dim,
                        pose_dim = self.pose_dim,
                        output_dim=self.bottleneck_size)
        self.decoder = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])
        self.decoder2 = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])
        self.estimate = Estimator(bottleneck_size=3 + self.bottleneck_size)
        self.estimate2 = Estimator(bottleneck_size=3+self.bottleneck_size)
        self.refine = Refiner(bottleneck_size=3 + self.bottleneck_size)

    def forward(self,x,points,vector1=0,vector2=0,mode='deform1'):
        
        x = self.encoder(x)
        if points.size(1) != 3:
            points = points.transpose(2,1)
        y = x.unsqueeze(2).expand(x.size(0), x.size(1), points.size(2)).contiguous()
        y = torch.cat((points, y), 1).contiguous()
        if mode == 'deform1':
            outs = self.decoder[0](y)
        elif mode == 'deform2':
            outs = self.decoder2[0](y)
            outs = outs + points
        elif mode == 'estimate':
            outs = self.estimate(y)
        elif mode == 'estimate2':
            outs = self.estimate2(y)
        elif mode == 'refine':
            outs = self.refine(y)
            outs1 = outs[:, 0].unsqueeze(1)
            outs2 = outs[:, 1].unsqueeze(1)
            outs = outs1 * vector1 + outs2 * vector2 + points
        else:
            outs = None
        return outs.contiguous().transpose(2,1).contiguous().squeeze(2)


class Pretrain(nn.Module):
    def __init__(self,  bottleneck_size = 1024,num_points=2500, class_dim =1, shape_dim = 5000 , pose_dim = 3):
        super(Pretrain, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.num_points = num_points
        self.pc_encoder = nn.Sequential(
        PointNetfeat(self.num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.class_dim = class_dim
        self.shape_dim = shape_dim
        self.pose_dim = pose_dim
        self.encoder = EmbeddingNet(class_dim = self.class_dim, 
                        shape_dim = self.shape_dim,
                        pose_dim = self.pose_dim,
                        output_dim=self.bottleneck_size)
        self.decoder = nn.ModuleList([DeformNet(bottleneck_size=3 + self.bottleneck_size)])

    def forward(self, x, mode='point'):
        if mode == 'point':
            x = self.pc_encoder(x)
        else:
            x = self.encoder(x)
        rand_grid = torch.cuda.FloatTensor(x.size(0),3,self.num_points)
        rand_grid.data.normal_(0,1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True))\
            .expand(x.size(0),3,self.num_points)
        y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
        y = torch.cat( (rand_grid, y), 1).contiguous()
        outs = self.decoder[0](y)
        return outs.contiguous().transpose(2,1).contiguous()

class BasicEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim=1024, hidden_dim=128):
        super(BasicEmbedding, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x 

class EmbeddingNet(nn.Module):

    def __init__(self, class_dim = 1, shape_dim = 5000, pose_dim = 3, output_dim = 1024):
        super(EmbeddingNet, self).__init__()
        self.class_embedding = BasicEmbedding( class_dim, output_dim)
        self.shape_embedding = BasicEmbedding( shape_dim, output_dim)
        self.pose_embedding  = BasicEmbedding( pose_dim, output_dim)
        self.linear          = nn.Linear(3*output_dim, output_dim)
        self.bn              = torch.nn.BatchNorm1d(output_dim)

    def forward(self, x ): 
        class_vec, shape_vec, pose_vec = x[:,0].unsqueeze(1), x[:,1:5001],  x[:,5001:]
        class_embed = self.class_embedding(class_vec)
        shape_embed = self.shape_embedding(shape_vec)
        pose_embed  = self.pose_embedding(pose_vec)
        feature_vec = torch.cat([class_embed, shape_embed, pose_embed], 1)
        feature_vec = self.linear(feature_vec)
        feature_vec = self.bn(feature_vec)
        return feature_vec

if __name__ =="__main__":
    
    batch = 3
    class_vec = torch.randn(batch,1).cuda()
    shape_vec = torch.randn(batch,5000).cuda()
    pose_vec  = torch.randn(batch,3).cuda()
    svr_tmnet  = SVR_TMNet( bottleneck_size = 1024, class_dim = 1,  shape_dim = 5000 , pose_dim = 3).cuda()
    svr_tmnet.eval()
    input = torch.cat([class_vec, shape_vec, pose_vec], dim = 1).float().cuda() 
    points = torch.randn(batch, 30000, 3).cuda()
    points = points.transpose(2, 1).contiguous()
    normals = torch.randn(batch, 10000, 3).cuda()    
    feature_vec = svr_tmnet(input, points,mode='deform1') 
    