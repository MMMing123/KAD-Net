import tqdm
import torch
from dataloader import *
from torch.utils.data import Dataset,DataLoader
from utils.utils import model_builder, Normalize_depth
from utils.HandposeEvaluation import *
import pickle
import tqdm
import sys
import os
import time
import argparse
import yaml
from utils.forwardpass import get_EvalFunction

import re
def getNumber(s):
    return int(re.findall('[0-9]+', s)[0])


###############################################
# print_detail_uvd: bool, whether to print the average error in each direction of UVD
@torch.no_grad()
def evaluate(model_path, model_name, loader, args, test_set , print_detail_crop=True,print_detail_uvd=False,print_detail_xyz=False):
    
    ckpt_state = torch.load(model_path, weights_only=False)["model"]
    # Remap old checkpoint keys: "hg.hg." -> "hg.0.hg." to match current nn.ModuleList structure
    remapped_state = {}
    for k, v in ckpt_state.items():
        if k.startswith("hg.hg."):
            new_key = "hg.0.hg." + k[len("hg.hg."):]
        else:
            new_key = k
        remapped_state[new_key] = v
    model.load_state_dict(remapped_state, strict=False)
    model.eval()
    
    GT_crop, GT_UVD_orig, GT_3D_orig, GT_matrix, estimation_cropped= [], [], [], [], []
    
    loop = tqdm.tqdm(loader)#preds=[[],[],[],[]] args.joint_dim

    for i, data in enumerate(loop):
        loop.set_description(model_name)

        inputs, gt2Dcrop, gt2Dorignal, gt3Dorignal, com, M_inv, cubesize = \
            data[0].to(device),data[1].to(device),data[2].to(device),data[3].to(device),\
            data[4].to(device),data[5].to(device),data[6].to(device)
        
        outputs,_,_,_,_,_,_,_  = model(inputs)

        ## shape is (bs,keypoints,3)
        # Denormalize the D in the predicted UVD and convert back to global absolute coordinates
        preds = Normalize_depth(outputs,cubesize,com,add_com=True)
        # Denormalize the D in the "detected" 2D UVD coordinates and convert back to global absolute coordinates
        gt_crop = Normalize_depth(gt2Dcrop,cubesize,com,add_com=True)

        GT_crop.append(gt_crop)
        GT_UVD_orig.append(gt2Dorignal)
        GT_3D_orig.append(gt3Dorignal)
        GT_matrix.append(M_inv)
        estimation_cropped.append(preds)

    ## Concatenate along the first dimension by default
    GT_crop=torch.cat(GT_crop).cpu() # (B,K,joint_dim)
    GT_UVD_orig=torch.cat(GT_UVD_orig).cpu() # (B,K,joint_dim)
    GT_3D_orig=torch.cat(GT_3D_orig).cpu() # (B,K,joint_dim)
    GT_matrix=torch.cat(GT_matrix).cpu() # (B,3,3)
    estimation_cropped=torch.cat(estimation_cropped).cpu() # (B,K,joint_dim)

    ## Evaluator
    # Args: 1. predicted 2D joints uvd (cropped) 2. ground-truth 2D joints uvd (cropped)
    Evaluator=HandposeEvaluation(estimation_cropped,GT_crop)

    output_message=f'--------------------- {model_name} ------------------------\n'
    
    # Cropped UVD
    if print_detail_crop:    
        res=Evaluator.getErrorPerDimension(printOut=False)
        output_message=output_message+"\nUVD_Cropped:\n"+res+"\n###############################\n"
        
    output_message=output_message+f"\nThe error UVD in the cropped version={Evaluator.getMeanError():.3f}\n"+"###############################\n"
    
    # Original UVD: convert back to the original UVD
    prediction_UVDorig=CropToOriginal(estimation_cropped,GT_matrix.float())
    del estimation_cropped,GT_matrix
    Evaluator.update(prediction_UVDorig,GT_UVD_orig)
    #prediction_ArrayToFile(prediction_UVDorig.numpy(),"me.txt");
    
    if print_detail_uvd:    
        res=Evaluator.getErrorPerDimension(printOut=False)
        output_message=output_message+"\nUVD_Original:\n"+res+"\n###############################\n"
        
    output_message=output_message+f"\nThe error in Original UVD ={Evaluator.getMeanError():.3f}\n"+"###############################\n"

    # 3D XYZ
    estimation_xyz=test_set.convert_uvd_to_xyz_tensor( prediction_UVDorig )
    Evaluator.update(estimation_xyz,GT_3D_orig)
 
    if print_detail_xyz:    
        res=Evaluator.getErrorPerDimension(printOut=False)
        output_message=output_message+ "\n3D results:\n" + res +"\n###############################\n"


    final_3Derror=Evaluator.getMeanError()
        
    output_message=output_message+f"\nFinal 3D error results: {final_3Derror:.3f}\n\n"+ 100*"=" + "\n"

    
    if args.save_results:
        f= open("results.txt","a+")
        f.write(output_message)
        f.close()
    else:
        print(output_message)

    return final_3Derror, prediction_UVDorig
    


####### MAIN LOOP #############

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--batch_size', default=16,type=int, help="batch_size")
    parser.add_argument('--cuda_id', default=-1, type=int, help="Cuda ID")
    # Model path
    parser.add_argument('--path', default="./icvl_experiment/checkpoints", type=str, help="the address of the dataset")
    parser.add_argument('--num_workers', default=4, type=int, help="num of subprocesses for data loading")
    parser.add_argument('--joint_dim', default=3, type=int, help="determine if it is 3D or 2D")
    parser.add_argument('--save_results', default=1, type=int, help="determine if the the results are written into a file")
    parser.add_argument('--center_refined', default=1, type=int, help="determine if the the results are written into a file")
    parser.add_argument('--dataset',default="icvl", choices=('nyu', 'icvl','msra'),type=str,help="which dataset to use")
    parser.add_argument('--save_preds',default="none", choices=('none', 'best','last'),type=str,help="Which model checkpoint to select to sotre its prediction")
    parser.add_argument('--pred_file_name',default="preds.txt", type=str,help="The name of the preds filename")
    parser.add_argument('--num_clean', default=1, type=int, help="if turned on, only the checkpoint corresponding to the best and last num_clean epochs will be kept")

    args = parser.parse_args()
    list_files=os.listdir(args.path)
    list_files.sort(key=getNumber) # Sort by number
    
    
    model_path=os.path.join(args.path,list_files[-1])
    setting=torch.load(model_path, weights_only=False)["args"]
    args.dataset=setting.dataset

    # Load datasetpath from the corresponding YAML config file
    config_file = os.path.join("configs", f"{args.dataset}.yaml")
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    datasetpath = yaml_config["datasetpath"]

    if args.dataset=="nyu":
        print("NYU dataset will be used")
        test_set=NYUHandPoseDataset(train=False,basepath=datasetpath,center_refined=args.center_refined)
        
    elif args.dataset=="icvl":
        print("ICVL dataset will be used")
        test_set=ICVLHandPoseDataset(train=False,basepath=datasetpath,center_refined=args.center_refined)

    elif args.dataset=="msra":
        print("MSRA dataset will be used")
        test_set=MSRAHandPoseDataset(train=False,basepath=datasetpath, LeaveOut_subject=setting.leaveout_subject , use_default_cube=setting.use_default_cube)

    if args.cuda_id == -1:
        default_cuda_id = "cuda:{}".format(int(setting.default_cuda_id))
    else:
        default_cuda_id = "cuda:{}".format(args.cuda_id)
        
        
    device = torch.device(default_cuda_id if torch.cuda.is_available() else "cpu")
    
    model = model_builder(setting.model_name,num_joints=test_set.num_joints, args=setting).to(device)


    names, errors3D = [], []

    best_prediction_UVD = None

    print("Initialization Done, Ready to start evaluationg...\n")

    for file in list_files:
            model_path=os.path.join(args.path,file)
            setting=torch.load(model_path, weights_only=False)["args"] # Load corresponding training arguments

            ## Set the corresponding 2D/3D crop area
            test_set.cropSize = (setting.cropSize,setting.cropSize)
           # test_set.cropSize3D = [setting.cubic_size,setting.cubic_size,setting.cubic_size]

            if args.dataset=="nyu":
                test_set.camID = setting.camid
                test_set.doLoadRealSample = ()
            
            testloader = DataLoader(test_set , batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, pin_memory = True)
            # Evaluate
            final_3Derror, prediction_UVDorig = \
                evaluate(model_path,file,testloader,args,test_set,
                         print_detail_crop=True,print_detail_uvd=True,print_detail_xyz=True)

            if len(errors3D)>0:
                if final_3Derror < np.min(errors3D):
                    best_prediction_UVD = prediction_UVDorig
            else:
                best_prediction_UVD = prediction_UVDorig

            names.append(file)
            errors3D.append(final_3Derror)


    best3Derror_indx = np.argmin(errors3D)
    best_3Derr = errors3D[best3Derror_indx]
    best_name = names[best3Derror_indx]
    output_message = "\n"+"###"*10+"\n"+f"Best Model = {best_name} => 3D error = {best_3Derr:.3f}\n\n--"
    f= open("results.txt","a+")
    f.write(output_message)
    f.close()
    print(best_name)

    if args.save_preds != "none":
        addr = os.path.join("".join([s+"/" for s in args.path.split("/")[:-1]]) , args.pred_file_name)
        if args.save_preds == "best":
            prediction_ArrayToFile(best_prediction_UVD, addr)

        if args.save_preds == "last":
            prediction_ArrayToFile(prediction_UVDorig, addr)


    if args.num_clean != -1:
        for i in range(len(names)- args.num_clean):
            if names[i] != best_name:
                os.remove(os.path.join(args.path,names[i]))
