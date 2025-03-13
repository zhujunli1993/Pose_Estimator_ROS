import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from config.config import *
from .Cross_Atten import CrossAttention
FLAGS = flags.FLAGS


class Rot_green(nn.Module):
    def __init__(self):
        super(Rot_green, self).__init__()
 
        self.conv_clip = torch.nn.Conv1d(512, 256, 1)
        self.cross = CrossAttention(dim=256, heads=2)
        self.conv1 = torch.nn.Conv1d(1286, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 4, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        
    def forward(self, x, clip_feat_r=None):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0] 
        
        clip_feat_r = torch.unsqueeze(clip_feat_r, dim=-1)
        clip_feat_r = self.conv_clip(clip_feat_r)
         
        x = x.permute(0, 2, 1)
        clip_feat_r = clip_feat_r.permute(0, 2, 1)
        x = self.cross(x_kv=x, x_q=clip_feat_r)
        
        x = x.permute(0, 2, 1) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x


class Rot_red(nn.Module):
    def __init__(self):
        super(Rot_red, self).__init__()

        
        self.conv_clip = torch.nn.Conv1d(512, 256, 1)
        self.cross = CrossAttention(dim=256, heads=2)
        self.conv1 = torch.nn.Conv1d(1286, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 4, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        
    def forward(self, x, clip_feat_r=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0] 
        
        clip_feat_r = torch.unsqueeze(clip_feat_r, dim=-1)
        clip_feat_r = self.conv_clip(clip_feat_r)
         
        x = x.permute(0, 2, 1)
        clip_feat_r = clip_feat_r.permute(0, 2, 1)
        x = self.cross(x_kv=x, x_q=clip_feat_r)
        
        x = x.permute(0, 2, 1) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x