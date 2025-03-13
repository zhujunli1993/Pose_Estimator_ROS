import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
import numpy as np
import torch.nn.functional as F
from network.fs_net_repo.PoseR import Rot_red, Rot_green
from network.fs_net_repo.PoseTs import Pose_Ts
from network.fs_net_repo.FaceRecon import FaceRecon

FLAGS = flags.FLAGS

class PoseNet9D(nn.Module):
    def __init__(self):
        super(PoseNet9D, self).__init__()
        # Used the fsnet rot_green and rot_red directly
        self.rot_green = Rot_green() 
        self.rot_red = Rot_red()
               
        self.face_recon = FaceRecon()
        
        self.ts = Pose_Ts()
        

    def forward(self, points, obj_id, clip_r_feat, clip_t_feat):
        bs, p_num = points.shape[0], points.shape[1]

        recon, _, feat= self.face_recon(points - points.mean(dim=1, keepdim=True), obj_id,
                                            clip_r_feat, clip_t_feat)

        face_normal, face_dis, face_f, recon = [None]*4
    
        #  rotation
        
        green_R_vec = self.rot_green(feat.permute(0, 2, 1), clip_r_feat)
        red_R_vec = self.rot_red(feat.permute(0, 2, 1), clip_r_feat)
        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])

        # translation and size
        
        feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2) #(bs, 1024, 1923)
        T, s = self.ts(feat_for_ts.permute(0,2,1), clip_t_feat)
        Pred_T = T + points.mean(dim=1)  # bs x 3
        Pred_s = s  # this s is not the object size, it is the residual

        
        return recon, face_normal, face_dis, face_f, p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s



