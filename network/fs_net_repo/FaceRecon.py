# Modified from FS-Net
import torch.nn as nn
import network.fs_net_repo.gcn3d as gcn3d
import torch
import torch.nn.functional as F
from absl import app
import absl.flags as flags

FLAGS = flags.FLAGS


class FaceRecon(nn.Module):
    def __init__(self):
        super(FaceRecon, self).__init__()
        self.neighbor_num = FLAGS.gcn_n_num
        self.support_num = FLAGS.gcn_sup_num

        # 3D convolution for point cloud
        self.conv_0 = gcn3d.HSlayer_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = gcn3d.HS_layer(128, 128, support_num=self.support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.HS_layer(128, 256, support_num=self.support_num)
        self.conv_3 = gcn3d.HS_layer(256, 256, support_num=self.support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.HS_layer(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.recon_num = 3
        self.face_recon_num = FLAGS.face_recon_c

        

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)",
                clip_r_feat: "tensor (bs, 512)", 
                clip_t_feat: "tensor (bs, 1024)"
                ):
        """
        Return: (bs, vertice_num, class_num)
        """
        if len(vertices.shape) == 2:
            vertice_num, _ = vertices.size()
        else:
            bs, vertice_num, _ = vertices.size()
        # cat_id to one-hot
        
        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)

        one_hot = torch.zeros(bs, FLAGS.obj_c).to(cat_id.device).scatter_(1, obj_idh.long(), 1)
        # bs x verticenum x 6

        # ss = time.time()
        fm_0 = F.relu(self.conv_0(vertices, self.neighbor_num), inplace=True)
        fm_1 = F.relu(self.bn1(self.conv_1(vertices, fm_0, self.neighbor_num).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        fm_2 = F.relu(self.bn2(self.conv_2(v_pool_1, fm_pool_1, 
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(v_pool_1, fm_2, 
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2, min(self.neighbor_num, v_pool_2.shape[1] // 8))

        
        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor_new(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor_new(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor_new(fm_4, nearest_pool_2).squeeze(2)
        
        one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)  # (bs, vertice_num, cat_one_hot)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, one_hot], dim=2)
        
        '''
        feat_face = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
        feat_face = torch.mean(feat_face, dim=1, keepdim=True)  # bs x 1 x channel
        feat_face_re = feat_face.repeat(1, feat.shape[1], 1)
        '''

        recon, face = None, None
        return recon, face, feat


