import torch
import torch.nn as nn
from timm.layers import DropPath

from model_factory.Transformer import Mlp

# X_2 joints with 3 degrees of freedom
thumb_3joints = [1,2,3]
index_3joints = [4,5,6]
middle_3joints = [7,8,9]
ring_3joints = [10,11,12]
pinky_3joints = [13,14,15]
part_3joints = [thumb_3joints,index_3joints,middle_3joints,ring_3joints,pinky_3joints]


class TC(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """Inter-Part Convolution Module (IPC)

        Parameters:
            dim (int): Input feature dimension
            mlp_hidden_dim (int): MLP hidden layer dimension
            drop (float): Dropout rate for MLP layer, default 0
            drop_path (float): DropPath dropout rate, default 0
            act_layer: Activation function, default GELU
            norm_layer: Normalization layer, default LayerNorm
        """
        super().__init__()
        # Initialize DropPath module (stochastic depth regularization)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Define limb joint indices (example values, should actually be defined based on specific skeleton structure)
        self.index_1 = [1, 2, 3, 4, 5, 6,7,8,9, 10, 11, 12, 13, 14, 15]  # 5 limbs with 3 joints each

        # Level 1 convolution processing (3-joint limbs)
        self.norm_conv1 = norm_layer(dim)  # Input normalization
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, stride=3)  # Downsampling conv (kernel=3, stride=3)
        self.norm_conv1_mlp = norm_layer(dim)
        self.mlp_down_1 = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)  # Explicitly specify parameter names
        #
        # Level 2 convolution processing (2-joint limbs)
        # self.norm_conv2 = norm_layer(dim)  # Input normalization
        # self.conv2 = nn.Conv1d(dim, dim, kernel_size=2, stride=2)  # Downsampling conv (kernel=2, stride=2)
        # self.norm_conv2_mlp = norm_layer(dim)
        # self.mlp_down_2 = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gelu = nn.GELU()  # GELU activation function

    def forward(self, x_gcn):
        """Forward propagation

        Parameters:
            x_gcn (Tensor): Features from graph convolutional network [B, L, C]
            x_conv (Tensor): Features from regular convolution [B, L, C]

        Returns:
            Tensor: Enhanced features [B, L, C]
        """
        # Residual connection with initial features
        x_conv = x_gcn

        # ------------------- Level 1 processing (3-joint limbs) -------------------
        # Normalization processing
        x_conv_1 = self.norm_conv1(x_conv)  # [B, L, C]
        x_conv_1 = x_conv_1.permute(0, 2, 1)  # [B, C, L]

        # Select 3-joint limb indices
        x_pooling_1 = x_conv_1[:, :, self.index_1]  # [B, C, 12]
        # Convolution downsampling (12 -> 4 feature points)
        x_pooling_1 = self.conv1(x_pooling_1)  # [B, C, 4]
        x_pooling_1 = self.gelu(x_pooling_1)
        x_pooling_1 = self.drop_path(x_pooling_1)  # Apply DropPath

        # MLP processing
        x_pooling_1 = x_pooling_1.permute(0, 2, 1)  # [B, 4, C]
        x_pooling_1 = self.norm_conv1_mlp(x_pooling_1)
        x_pooling_1 = self.mlp_down_1(x_pooling_1)  # MLP processing
        x_pooling_1 = self.drop_path(x_pooling_1)  # Apply DropPath

        # Feature broadcast back to original dimension
        x_pooling_1 = x_pooling_1.permute(0, 2, 1)  # [B, C, 4]
        for i in range(len(part_3joints)):
            # Get number of joints for current limb (e.g., [0,1,2] needs to broadcast to last 2 joints)
            num_joints = len(part_3joints[i]) - 1
            # Broadcast pooled features to corresponding joints
            x_conv_1[:, :, part_3joints[i][1:]] = x_pooling_1[:, :, i].unsqueeze(-1).repeat(1, 1, num_joints).to(dtype=x_conv_1.dtype)

        x_conv_1 = x_conv_1.permute(0, 2, 1)  # Restore dimension [B, L, C]



        # Residual connection to integrate all features
        x_conv = x_conv_1  + x_conv
        return x_conv

if __name__ == '__main__':
    input = torch.randn((32,16,128))

    # Suggested parameter configuration
    dim = 128          # Input feature dimension consistent with input channel C=256
    mlp_hidden_dim = 512  # MLP hidden layer dimension, usually 2-4 times input dimension (512 or 1024)
    drop = 0.1         # Moderate overfitting prevention
    drop_path = 0.1    # Consistent with drop value, maintain regularization strength
    # Module initialization example
    tc = TC(dim=128, mlp_hidden_dim=512,drop=0.1,drop_path=0.1)

    output = tc(input,input)
    print(output.shape)