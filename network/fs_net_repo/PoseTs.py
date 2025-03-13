import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from .Cross_Atten import CrossAttention
FLAGS = flags.FLAGS

# Point_center  encode the segmented point cloud
# one more conv layer compared to original paper

class Pose_Ts(nn.Module):
    def __init__(self):
        super(Pose_Ts, self).__init__()

        self.cross = CrossAttention(dim=256, heads=2)
        self.conv_clip = torch.nn.Conv1d(1024, 256, 1)
        self.conv1 = torch.nn.Conv1d(1289, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 6, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
    def forward(self, x, clip_feat_t=None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        
        clip_feat_t = torch.unsqueeze(clip_feat_t, dim=-1)
        clip_feat_t = self.conv_clip(clip_feat_t)
            
        x = x.permute(0, 2, 1)
        clip_feat_t = clip_feat_t.permute(0, 2, 1)
        x = self.cross(x_kv=x, x_q=clip_feat_t)  
         
        x = x.permute(0, 2, 1)  
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs