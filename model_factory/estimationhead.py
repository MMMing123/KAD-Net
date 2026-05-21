import torch
import torch.nn as nn
import numpy as np


## Concatenate feature map height and width by row
## Generate a tensor, shape[1, 1, 4096], values 64*(0-63)
from model_factory.Graph_base import Graph_Conv


def GetValuesX(dimension=64):
    n = dimension
    vec = np.linspace(0, dimension - 1, dimension).reshape(1, -1)
    Xs = np.linspace(0, dimension - 1, dimension).reshape(1, -1)
    for i in range(n - 1):
        Xs = np.concatenate([Xs, vec], axis=1)
    Xs = np.float32(np.expand_dims(Xs, axis=0))

    return torch.from_numpy(Xs)

## Generate a tensor, shape[1, 1, 4096], values 64*(0)->64*(1)->64*(2)->....64*(63)
def GetValuesY(dimension=64):
    res = np.zeros((1, dimension * dimension))
    for i in range(dimension):
        res[0, (i * dimension):((i + 1) * dimension)] = i
    res = np.float32(np.expand_dims(res, axis=0))
    return torch.from_numpy(res)

## Spatial softmax [bs,14,64,64]->[bs,14,64,64]
# Pass in train_spread=False; num_channel=14, spread=None
# spread is used to control the "sharpness" or "smoothness" of Softmax application in spatial dimension for each channel, similar to, a tensor with shape (1,num_chanel,1): for multiplication
# Pass in train_spread, boolean type, represents whether to train the above parameters, defaults to False, set to 1
class AdaptiveSpatialSoftmaxLayer(nn.Module):
    def __init__(self, spread=None, train_spread=False, num_channel=14):
        super(AdaptiveSpatialSoftmaxLayer, self).__init__()
        # spread should be a torch tensor of size (1,num_chanel,1)
        # the softmax is applied over spatial dimensions
        # train determines whether you would like to train the spread parameters as well
        if spread is None:
            self.spread = nn.Parameter(torch.ones(1, num_channel, 1))
        else:
            self.spread = nn.Parameter(spread)

        self.spread.requires_grad = bool(train_spread)

    # x(batch,num_channel,height,width)
    def forward(self, x):
        SpacialSoftmax = nn.Softmax(dim=2) # Create spatial softmax layer
        num_batch = x.shape[0]
        num_channel = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        inp = x.view(num_batch, num_channel, -1) # [bs,chanel,w*h]
        # if self.spread is not None:
        res = torch.mul(inp, self.spread)
        res = SpacialSoftmax(res) # Perform spatial softmax

        return res.reshape(num_batch, num_channel, height, width)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, BN, num_G, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes) if BN else nn.GroupNorm(num_G, inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes) if BN else nn.GroupNorm(num_G, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes) if BN else nn.GroupNorm(num_G, planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

# Default initialization parameters:
# 1. num_blocks: 3 number of residual blocks
# 2. block: Bottleneck residual block constructor
# 3. input_dim: 256 input dimension, here input refers to channel number after HourglassNet feature extraction
# 4. depth_dim: 64 depth dimension D, as stated in paper set to 64
class EstimationHead(nn.Module):
    def __init__(self, num_blocks, block, input_dim=256, depth_dim=64, num_classes=14, attention_num_blocks=3,
                 train_spread=False, BN=True, num_G=16):
        super(EstimationHead, self).__init__()

        self.num_feats = int(input_dim / 2)
        self.input_dim = input_dim # Input dimension
        self.depth_dim = depth_dim # This is D in the paper
        self.num_classes = num_classes

        ch = self.num_feats * 2

        self.BN = BN
        self.relu = nn.ReLU(inplace=True)

        self.soft = AdaptiveSpatialSoftmaxLayer(train_spread=train_spread, num_channel=num_classes)

        ## Register a buffer in nn.Module, variables inside won't be trained
        self.register_buffer("Xs", GetValuesX())
        self.register_buffer("Ys", GetValuesY())

        ## UV branch has three parts: 1. Three residuals: no shape change 2. 1*1 conv: no shape change
        # 3. 1*1 conv: convolve 256 channels to KEYPOINTS channels, no width-height change  
        # Total: [bs,256,64,64]->[bs,num_classes,64,64]
        # input_dim = 256 num_feats：256/2
        # ch = 256
        ## UV-GRM
        self.GRM = Graph_Conv(256, 16)
        self.UVbranch_pre = nn.Sequential(self._make_residual(block, input_dim, self.num_feats, num_blocks, BN, num_G),
                                          self._make_fc(ch, ch))
        self.UVbranch_finally = nn.Conv2d(ch, num_classes, kernel_size=1, bias=True)

        ## Attention fusion parameters: trainable, default value: 0.5 [1,keypoints,1,1]
        self.Betas = torch.nn.Parameter(torch.ones(1, num_classes, 1, 1) * 0.5)
        ## Depth branch also has three parts: 1. Three residuals: no shape change 2. 1*1 conv: no shape change
        # 3. 1*1 conv: convolve 256 channels to D channels, no width-height change
        # Total: [bs,256,64,64]->[bs,D,64,64]
        self.DepthBranch_1 = nn.Sequential(self._make_residual(block, input_dim, self.num_feats, num_blocks, BN, num_G),
                                         self._make_fc(ch, ch), nn.Conv2d(ch, self.depth_dim, kernel_size=1, bias=True))
        self.DepthBranch_2 = nn.Sequential(self._make_residual(block, input_dim, self.num_feats, num_blocks, BN, num_G),
                                         self._make_fc(ch, ch), nn.Conv2d(ch, self.depth_dim, kernel_size=1, bias=True))
        self.DepthBranch_3 = nn.Sequential(self._make_residual(block, input_dim, self.num_feats, num_blocks, BN, num_G),
                                         self._make_fc(ch, ch), nn.Conv2d(ch, self.depth_dim, kernel_size=1, bias=True))
        self.DepthBranch_4 = nn.Sequential(self._make_residual(block, input_dim, self.num_feats, num_blocks, BN, num_G),
                                         self._make_fc(ch, ch), nn.Conv2d(ch, self.depth_dim, kernel_size=1, bias=True))
        self.DepthBranch_5 = nn.Sequential(self._make_residual(block, input_dim, self.num_feats, num_blocks, BN, num_G),
                                         self._make_fc(ch, ch), nn.Conv2d(ch, self.depth_dim, kernel_size=1, bias=True))
        self.DepthBranch_6 = nn.Sequential(self._make_residual(block, input_dim, self.num_feats, num_blocks, BN, num_G),
                                         self._make_fc(ch, ch), nn.Conv2d(ch, self.depth_dim, kernel_size=1, bias=True))

        ## FC
        # First one
        self.FC_Palm_1 = nn.Linear(64,64)
        self.FC_Pinky_1 = nn.Linear(192,192)
        self.FC_Ring_1 = nn.Linear(192,192)
        self.FC_Middle_1 = nn.Linear(192,192)
        self.FC_Index_1 = nn.Linear(192,192)
        self.FC_Thumb_1 = nn.Linear(192,192)

        self.dropout = nn.Dropout(0.4)



        # Second one
        self.FC_Palm_2 = nn.Linear(64, 1)
        self.FC_Pinky_2= nn.Linear(192, 3)
        self.FC_Ring_2 = nn.Linear(192, 3)
        self.FC_Middle_2 = nn.Linear(192, 3)
        self.FC_Index_2 = nn.Linear(192, 3)
        self.FC_Thumb_2 = nn.Linear(192, 3)

        self.FC_Finally = nn.Linear(1024,16)

        ## Attention enhancement branch is exactly the same as UV branch
        self.AttentionEnhBranch = nn.Sequential(
            self._make_residual(block, input_dim, self.num_feats, attention_num_blocks, BN, num_G),
            self._make_fc(ch, ch), nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))



    ## Create consecutive residual blocks
    # inplanes: base channel number for middle layer. Actual output channel number is planes * block.expansion
    # blocks is the quantity
    def _make_residual(self, block, inplanes, planes, blocks, BN, num_G, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=True), )

        layers = []
        layers.append(block(inplanes, planes, BN, num_G, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, BN, num_G))

        return nn.Sequential(*layers)

    ## Create 1*1 conv layer (followed by BN and RELU)
    # Creation parameters: input conv channel number and output channel number
    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(outplanes) if self.BN else nn.GroupNorm(self.num_G, inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    # x[bs,256,64,64]
    def forward(self, x, return_heatmap=False, scale_factor=2):
        num_batch = x.shape[0]


        uv_out = self.UVbranch_pre(x) # [bs,256,64,64]->[bs,keypoints,64,64]
        ## After GRM
        uv_out,Joints = self.GRM(uv_out) #[bs,keypoints,64,64],[bs,[bs,keypoints,2]
        uv_out = self.UVbranch_finally(uv_out)
        hmp = self.soft(uv_out)  # Perform spatial softmax ->[bs,keypoints,64,64]

        ## Heatmap processing: weighted average
        X0 = torch.mul(hmp.view(num_batch, self.num_classes, -1), self.Xs) #[bs,keypoints,w*h]*[1,1,4096]=[bs,keypoints,w*h]
        X0 = torch.sum(X0, dim=-1) # Sum to get value ->[bs,keypoints]
        Y0 = torch.mul(hmp.view(num_batch, self.num_classes, -1), self.Ys)
        Y0 = torch.sum(Y0, dim=-1)
        X0 = torch.unsqueeze(X0, dim=-1) # Insert dimension ->[bs,keypoints,1]
        Y0 = torch.unsqueeze(Y0, dim=-1)
        UV0 = torch.cat((X0, Y0), dim=-1) # Concatenate ->[bs,keypoints,2]

        ## Attention enhancement branch
        aux_attention = self.AttentionEnhBranch(x) # [bs,256,64,64]->[bs,keypoints,64,64]

        ## Attention fusion, NOTE: fusing the variable just out of UV branch
        # [1,keypoints,1,1]*[bs,keypoints,64,64]
        # Then perform spatial softmax->[bs,keypoints,64,64]
        attentionmap = self.soft(self.Betas * aux_attention + (1 - self.Betas) * uv_out)


        ## Background
        att_Plam = attentionmap[:,0:1,:,:]
        att_Thumb = attentionmap[:,1:4,:,:]
        att_Index = attentionmap[:,4:7,:,:]
        att_Middle = attentionmap[:,7:10,:,:]
        att_Ring = attentionmap[:,10:13,:,:]
        att_Pinky = attentionmap[:,13:16,:,:]


        ## Depth branch multi-task
        d1 = self.DepthBranch_1(x) # [bs,256,64,64]->[bs,D,64,64] D=64
        d2 = self.DepthBranch_2(x) # [bs,256,64,64]->[bs,D,64,64] D=64
        d3 = self.DepthBranch_3(x) # [bs,256,64,64]->[bs,D,64,64] D=64
        d4 = self.DepthBranch_4(x) # [bs,256,64,64]->[bs,D,64,64] D=64
        d5 = self.DepthBranch_5(x) # [bs,256,64,64]->[bs,D,64,64] D=64
        d6 = self.DepthBranch_6(x) # [bs,256,64,64]->[bs,D,64,64] D=64


        ## Guidance
        # f1_vector[bs,1,128] f2-f6_vector[bs,3,128]
        # att_Palm[bs,1,64,64] -->     unsqueeze:[bs,1,1,64,64]
        # f1_resnet2[bs,128,64,64] --> unsqueeze:[bs,1,128,64,64]
        # att_Pinky[bs,3,64,64] -->     unsqueeze(2):[bs,3,1,64,64]
        # f2_resnet2[bs,128,64,64] --> unsqueeze(1):[bs,1,128,64,64]
        f1_vector = torch.sum(att_Plam.unsqueeze(2)*d1.unsqueeze(1),dim=(-1,-2))
        f2_vector = torch.sum(att_Pinky.unsqueeze(2)*d2.unsqueeze(1),dim=(-1,-2))
        f3_vector = torch.sum(att_Ring.unsqueeze(2)*d3.unsqueeze(1),dim=(-1,-2))
        f4_vector = torch.sum(att_Middle.unsqueeze(2)*d4.unsqueeze(1),dim=(-1,-2))
        f5_vector = torch.sum(att_Index.unsqueeze(2)*d5.unsqueeze(1),dim=(-1,-2))
        f6_vector = torch.sum(att_Thumb.unsqueeze(2)*d6.unsqueeze(1),dim=(-1,-2))

        ## f1_vector[bs,128] f2-f6_vector[bs,384]
        f1_vector = f1_vector.view((num_batch,-1))
        f2_vector = f2_vector.view((num_batch,-1))
        f3_vector = f3_vector.view((num_batch,-1))
        f4_vector = f4_vector.view((num_batch,-1))
        f5_vector = f5_vector.view((num_batch,-1))
        f6_vector = f6_vector.view((num_batch,-1))

        ## First FC
        #  Then f1_fc1 [bs,1] f2_fc1-f6_fc1[bs,2]
        fc1 = self.FC_Palm_1(f1_vector)
        fc2 = self.FC_Pinky_1(f2_vector)
        fc3 = self.FC_Ring_1(f3_vector)
        fc4 = self.FC_Middle_1(f4_vector)
        fc5 = self.FC_Index_1(f5_vector)
        fc6 = self.FC_Thumb_1(f6_vector)

        fc1 = self.relu(fc1)
        fc2 = self.relu(fc2)
        fc3 = self.relu(fc3)
        fc4 = self.relu(fc4)
        fc5 = self.relu(fc5)
        fc6 = self.relu(fc6)

        fc1 =self.dropout(fc1)
        fc2 =self.dropout(fc2)
        fc3 =self.dropout(fc3)
        fc4 =self.dropout(fc4)
        fc5 =self.dropout(fc5)
        fc6 =self.dropout(fc6)

        ## Second FC
        #  Then f1_fc1 [bs,1] f2_fc1-f6_fc1[bs,2]
        fc1_2 = self.FC_Palm_2(fc1)
        fc2_2 = self.FC_Pinky_2(fc2)
        fc3_2 = self.FC_Ring_2(fc3)
        fc4_2 = self.FC_Middle_2(fc4)
        fc5_2 = self.FC_Index_2(fc5)
        fc6_2 = self.FC_Thumb_2(fc6)

        # shape[bs,1024]
        con_fc = torch.cat((fc1,fc2,fc3,fc4,fc5,fc6),dim=1)
        # shape[bs,keypoints]
        d_finally =  self.FC_Finally(con_fc)
        d_finally = torch.unsqueeze(d_finally,dim=2)


        ## Concatenate results # [1,16,3],f1_fc3 [bs,1]; f2_fc2-f6_fc2 [bs,3]
        UVD = torch.cat([UV0 * scale_factor, d_finally], dim=-1)

        return UVD,fc1_2,fc2_2,fc3_2,fc4_2,fc5_2,fc6_2,Joints


def build_EstimationHead(num_blocks, input_dim=256, depth_dim=64, num_classes=14, train_spread=False, BN=True,
                         num_G=16):
    return EstimationHead(num_blocks, Bottleneck, input_dim, depth_dim, num_classes, train_spread, BN, num_G)
