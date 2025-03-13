import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d
import sys
sys.path.append('..')


from tools.training_utils import get_gt_v
from .Rot_3DGC import Pts_3DGC

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.
    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return -(features[:, None, :] - features[None, :, :]).norm(2, dim=-1) # negative L2 norm
        elif self.similarity_type == 'cos':
            
            # make sure the last row is [0, 0, 0, 1]
            bs = features.shape[0]
            cos = nn.CosineSimilarity(dim=2, eps=1e-8)
            features_x1 = features.unsqueeze(1)  # bs*1*3
            features_x2 = features.unsqueeze(0)  # 1*bs*3
            
            return cos(features_x1, features_x2)
            
        else:
            raise ValueError(self.similarity_type)
class Projection(nn.Module):
    
    def __init__(
        self,
        pts_embedding
        ):
        super(Projection, self).__init__()
        self.projection_dim = pts_embedding
        self.w1 = nn.Linear(pts_embedding, pts_embedding, bias=False)
        self.bn1 = nn.BatchNorm1d(pts_embedding)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(pts_embedding, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)
    
    def forward(self, embedding):
        
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(embedding)))))
        

class Model_Rot_all(nn.Module):
    def __init__(
        self
    ):
        super(Model_Rot_all, self).__init__()
        ''' encode point clouds '''
        self.pts_encoder = Pts_3DGC()

        self.project_head = Projection(512)
        
        
    def forward(self, batch):
        
        bs = batch['zero_mean_pts_1'].shape[0]
        # Getting point cloud and gt pose Features
        
        pts_1_features = self.project_head(self.pts_encoder(batch['zero_mean_pts_1'])) #bs*N*3
        return pts_1_features

    

