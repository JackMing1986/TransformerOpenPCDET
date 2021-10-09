import math
import torch
from torch import nn

from .vfe_template import VFETemplate


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.w = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.w.size(2))
        for i in range(self.num_class):
            self.w[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.w * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x
    

class TransformerVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.feature_fc = nn.Linear(7, 16)
        self.pe = nn.Sequential(
            nn.Linear(3, 16 // 2),
            nn.ReLU(),
            nn.Linear(16 // 2, 16),
            nn.ReLU()
            )
        self.query_embed = nn.Embedding(64, 16)
        self.transformer = nn.Transformer(
            d_model=16,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=4,
            dim_feedforward=128)
        self.fc = GroupWiseLinear(64, 16, bias=True)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_point_features
    
    def get_voxel_features(self, voxel_features, voxel_num_points, coords):
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        xyzs = voxel_features[..., :3]
        pos_embeddings = self.pe(xyzs)
        features = [voxel_features[..., 3:], f_cluster, f_center]
        features = torch.cat(features, dim=-1)
        feat_embeddings = self.feature_fc(features)
        src_embedding = feat_embeddings + pos_embeddings
        bs, _, _ = src_embedding.shape
        src_embedding = src_embedding.transpose(0, 1)
        tgt_embedding = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        out_embeddings = self.transformer(src_embedding, tgt_embedding)

        return self.fc(out_embeddings.transpose(0, 1))


    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """

        batch_dict['voxel_features'] = self.get_voxel_features(
            batch_dict['voxels'], 
            batch_dict['voxel_num_points'], 
            batch_dict['voxel_coords']
        )
        batch_dict['voxel_features_ext'] = self.get_voxel_features(
            batch_dict['voxels_ext'], 
            batch_dict['voxel_num_points_ext'], 
            batch_dict['voxel_coords_ext']
        )

        return batch_dict
