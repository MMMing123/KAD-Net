import numpy as np
import torch
from torch import nn

from model_factory.TC import TC


class Graph_mapping(nn.Module):
    def __init__(self, dim, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.dim = dim
        self.att = nn.Conv2d(dim, num_joints, kernel_size=1)  # Attention
        self.conv_D = nn.Conv2d(dim, dim // 2, kernel_size=1)   # Node features
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        bs, c, h, w = x1.size()

        att_out = self.att(x1)  # [bs, num_joints, h, w]
        spatial_attention = att_out.view(bs, self.num_joints, -1) # [bs, num_joints=N, h*w]
        spatial_attention = self.softmax(spatial_attention) #  softmax

        D = self.conv_D(x1)  # [bs, dim//2, h, w] ==  [bs, D, h, w]  （D=dim/2=C'）
        D_flat = D.view(bs, self.dim // 2, -1)  #  [bs, D, h*w]

        # [N×HW] × [HW,D] = [N,D]
        node_features = torch.bmm(spatial_attention, D_flat.transpose(1, 2)) # [N,D]=[num_joints, input_features/2=D]

        return node_features,spatial_attention

class Message_Passing(nn.Module):
    def __init__(self, D): # Create parameters node feature dimension D, which is C'
        super().__init__()
        ## Basic graph convolution: linear layer
        self.Linear = nn.Linear(D,D,bias=True)
        self.relu = nn.ReLU()
        self.ipc = TC(dim=128, mlp_hidden_dim=256, drop=0.3, drop_path=0.3)

    def forward(self, node_feature): # node_feature[bs,N,D] also [bs,N,C']

        bs = node_feature.shape[0]
        adjacency_matrix = get_icvl_adjacency_matrix(bs).to(node_feature.device)

        ## 1. FC
        node_feature_update = self.Linear(node_feature)
        ## 2. Aggregate neighbor features
        # [N*N] ×[N*D] = [N*D]
        out = torch.bmm(adjacency_matrix.float(),node_feature_update)
        out = self.relu(out)

        output = self.ipc(out)  # First GCN, second original

        return output



class Graph_Conv(nn.Module):
    def __init__(self, dim,num_joints):
        super().__init__()
        self.Graph_mapping_module = Graph_mapping(dim=dim, num_joints=num_joints)
        self.Message_Passing_module = Message_Passing(int(dim/2))

        self.expand_dim_conv = nn.Conv2d(dim//2,dim,1,bias=True)
        self.conv_F = nn.Conv2d(dim*2,dim,1,bias=True)

        self.JointMapping =  JointPosePredictor(num_joints,int(dim/2))

    def forward(self, x): # x[bs,c,w,h]
        bs = x.shape[0]
        c = x.shape[1]
        D = c//2
        w = x.shape[2]
        h = x.shape[3]

        ## 1. Feature map mapping node_feature[bs,N,C'/D]  spatial_attention_softmax[bs,N,HW]
        node_feature,spatial_attention_softmax = self.Graph_mapping_module(x)
        Joints = self.JointMapping(node_feature)

        ## 2. Message passing
        node_feature_F = self.Message_Passing_module(node_feature) # [bs,N,C'/D]

        ## 3. Remap feature map
        # [bs,HW,N] * [bs,N*C'] = [bs,HW,C']
        feature_middle = torch.bmm(spatial_attention_softmax.transpose(1, 2),node_feature_F)
        feature_middle = feature_middle.view(bs,D,w,h) # [bs,C',w,h]

        # Expand dimension through convolution
        feature_middle_expand = self.expand_dim_conv(feature_middle) # [bs,C,w,h]
        # Residual-like concatenation # [bs,2C,H,W]
        feature_middle_concat = torch.concat((x,feature_middle_expand),dim=1) # [bs,2C,H,W]
        # Final channel reduction convolution [bs,C,H,W]
        feature_map = self.conv_F(feature_middle_concat)

        return feature_map,Joints



# Return adjacency matrix tensor [bs,N,N]
def get_icvl_adjacency_matrix(bs):
    num_joints = 16
    adj_matrix = np.zeros((num_joints, num_joints))

    # ICVL dataset actual joint structure (16 nodes)
    fingers = [
        [1, 2, 3],     # Thumb
        [4, 5, 6],     # Index finger
        [7, 8, 9],     # Middle finger
        [10, 11, 12],  # Ring finger
        [13, 14, 15]   # Pinky
    ]

    # Add self-connections (set before other connections)
    np.fill_diagonal(adj_matrix, 1)

    # Connect wrist to each finger root
    for finger in fingers:
        adj_matrix[0][finger[0]] = 1
        adj_matrix[finger[0]][0] = 1

    # Connect finger joints
    for finger in fingers:
        for i in range(len(finger)-1):
            from_node, to_node = finger[i], finger[i+1]
            adj_matrix[from_node][to_node] = 1
            adj_matrix[to_node][from_node] = 1

    adj_matrix = torch.tensor(adj_matrix)  # [N*N]
    adj_matrix = adj_matrix.unsqueeze(0).repeat(bs, 1, 1)  # Result shape [bs, N, N]

    return adj_matrix

class JointPosePredictor(nn.Module):
    def __init__(self, num_joints, feat_dim):
        """
        Args:
            num_joints (int): Number of joints J
            feat_dim (int): Input feature dimension D
        """
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=feat_dim,  # Input feature dimension D
            out_channels=2,  # 2D coordinate output
            kernel_size=1,  # 1D conv kernel
            bias=True
        )
        self.num_joints = num_joints

    def forward(self, x):
        """
        Args:
            J  == N
            x (Tensor): Input features [batch_size, J, D]
        Returns:
            Tensor: Predicted 3D coordinates [batch_size, J, 3]
        """
        # Dimension adjustment: [B, J, D] -> [B, D, J]
        x = x.permute(0, 2, 1)

        # 1D conv processing: [B, D, J] -> [B, 2, J]
        y = self.conv1d(x)

        # Dimension restoration: [B, 2, J] -> [B, J, 3]
        return y.permute(0, 2, 1)


if __name__ == '__main__':
    x1 = torch.randn(1, 256, 224, 224)
    model = Graph_Conv(256,16)
    out,joints = model(x1)

    print(out.shape)  # ([1, 256, 224, 224])
    print(joints.shape)  # ([1, 16, 2)




