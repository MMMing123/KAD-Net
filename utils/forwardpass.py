import torch
import os
import numpy as np
import time

from dataloader import denormalize_depth
from utils.utils import loss_masked, Normalize_depth

FORWARD_PASS = {'hglass': "normal"}

EVAL_FUNCTIONS = {'hglass': "normal"}

################ GETTERs ############################

def get_forwardPass(args):
    arc_name = args.model_name.split("_")[0].lower() # model_name defaults to hglass_2_1
    mode = FORWARD_PASS[arc_name]
    if mode == "normal":
        return Normal_forwardPass
    else:
        raise NotImplementedError





######################### Forward passes #######################################

## Normal forward pass
# Returns: loss (scalar) and related loss dictionary
def Normal_forwardPass(model, inputs, gt_uvd, lossFunction, cubesize,coms, joint_mask, visible_mask,suffix = None):
    mask = joint_mask * visible_mask # Boolean * number = element-wise multiplication, obtain visibility mask (may randomly drop joints)
    output,f1,f2,f3,f4,f5,f6,GRM_Joints = model(inputs) # Input shape [16,1,128,128] Output shape[16,14,3]
    # Denormalize D in output UVD
    output = Normalize_depth(output,sizes=cubesize,coms=coms,add_com=False)
    f1 = denormalize_depth(f1,sizes=cubesize,coms=coms,add_com=False)
    f2 = denormalize_depth(f2,sizes=cubesize,coms=coms,add_com=False)
    f3 = denormalize_depth(f3,sizes=cubesize,coms=coms,add_com=False)
    f4 = denormalize_depth(f4,sizes=cubesize,coms=coms,add_com=False)
    f5 = denormalize_depth(f5,sizes=cubesize,coms=coms,add_com=False)
    f6 = denormalize_depth(f6,sizes=cubesize,coms=coms,add_com=False)
    # Get average weighted loss (scalar, loss for each keypoint (average, excluding zeros)): torch.mean( lossFunction(output,gt_uvd)*mask )
    loss = loss_masked(output,GRM_Joints, gt_uvd, mask, lossFunction,f1,f2,f3,f4,f5,f6)
    loss_dict = {}
    suffix = ('' if suffix is None else suffix)
    # .detach() is a PyTorch method used to detach tensor from computation graph, thus stopping gradient tracking
    loss_dict[f"{suffix}Tot_loss"] = loss.detach()
    # loss_dict["visibility_rate"]: shape [keypointnums], represents visibility rate for each joint
    # visible_mask[bs,keypointnums,1]

    # Optional: whether to output visibility rate in log
    # loss_dict["visibility_rate"] = (torch.sum(visible_mask,dim=0)/visible_mask.shape[0]).squeeze()

    return loss, loss_dict



############################ Eval Functions ###################################################

def get_EvalFunction():
    return Normal_eval

def Normal_eval(inputs,outputs,cubesize,com,args):
    outputs[:,:,0:2]=outputs[:,:,0:2]
    return Normalize_depth(outputs,cubesize,com,add_com=True)
