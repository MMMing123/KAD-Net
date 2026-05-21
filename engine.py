import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.HandposeEvaluation import HandposeEvaluation
from utils.utils import *
import time
from tqdm import tqdm
from utils.utils import model_builder
from dataloader import DATASET_NUM_JOINTS
from utils.forwardpass import get_forwardPass


def Train_Test(model, data_loaders, args,lossFunction, optimizer, device, scheduler, fp16_scaler):

    writer = SummaryWriter("./Tensorboard")
    best_3Derror = 999
    best_epoch = 1

    # Get forward pass function object itself
    ForwardPassFunc = get_forwardPass(args)
    trainloader_labeled = data_loaders["trainloader_labeled"]
    # x  = torch.load("savedModel_E29_5.780.pt")["model"]
    # model.load_state_dict(x)
    for epoch in range(args.num_epoch):
        meter = AverageMeter(fmt=':.3f') # Class to calculate and store data

        #### Training ####
        model.train()
        start_time_iter = time.time()
        loop = tqdm(trainloader_labeled)
        loop.set_description(f"Epoch {epoch+1}: ") # Progress bar description information, i.e., the header

        total_loss = 0
        count = 0
        for i, data in enumerate(loop):

            inputs, gt_uvd, com, cubesize , joint_mask, visible_mask= \
                data[0].to(device), data[1].to(device), data[4].to(device), \
                data[6].to(device), data[7].to(device), data[8].to(device)

            # Denormalize D in UVD
            gt_uvd=Normalize_depth(gt_uvd,sizes=cubesize,coms=com,add_com=False).float()

            # forward + backward + optimize

            with torch.amp.autocast(device_type='cuda', enabled=(fp16_scaler is not None)):
                loss, loss_dict = ForwardPassFunc(model, inputs, gt_uvd, lossFunction, cubesize, com, joint_mask, visible_mask)

            total_loss += loss
            count += 1 # Equivalent to number of batches
            meter.update(loss_dict) # Update "data class" with loss dictionary, updates cumulative data in data dictionary, and updates counts and averages, calculates current average for each metric.

            # Update parameters
            Grad_Updater(loss, model, optimizer, fp16_scaler, args)

        # str(meter) will output the average of each metric according to format
        message=f"End of epoch: {epoch+1}: " + str(meter) + f" | Total Time: {(time.time()-start_time_iter)/60:.2f} mins"
        print(message)
        writer.add_scalar("train_loss", total_loss/count, epoch + 1)  # Record per batch


        if args.scheduler == "auto":
            scheduler.step( meter.averages["Tot_loss"] )
        else:
            scheduler.step()



        #### Evaluation ####
        # 1. Load model
        if args.dataset == "nyu":
            #print("NYU dataset will be used")
            os.environ['NYU_PATH'] = "../../../../e/D_DataSets/nyu/nyu_hand_dataset_v2/dataset"
            test_set = NYUHandPoseDataset(train=False, basepath=os.environ.get('NYU_PATH'),
                                          center_refined=args.center_refined,cropSize3D=[args.cubic_size,args.cubic_size,args.cubic_size])
        elif args.dataset == "icvl":
            #print("ICVL dataset will be used")
            test_set = ICVLHandPoseDataset(train=False, basepath=args.datasetpath,
                                           center_refined=args.center_refined)
        elif args.dataset == "msra":
            print("MSRA dataset will be used")
            test_set = MSRAHandPoseDataset(train=False, basepath=os.environ.get('MSRA_PATH'),
                                           LeaveOut_subject=args.leaveout_subject,
                                           use_default_cube=args.use_default_cube)

        testloader = DataLoader(test_set, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## 2. Evaluate
        model.eval()

        with torch.no_grad():
            GT_crop, GT_UVD_orig, GT_3D_orig, GT_matrix, estimation_cropped = [], [], [], [], []

            loop = tqdm(testloader)  # preds=[[],[],[],[]] args.joint_dim

            total_loss = 0
            count = 0
            for i, data in enumerate(loop):
                loop.set_description("test")

                inputs, gt2Dcrop, gt2Dorignal, gt3Dorignal, com, M_inv, cubesize = \
                    data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), \
                    data[4].to(device), data[5].to(device), data[6].to(device)

                outputs,_,_,_,_,_,_,_ = model(inputs)

                ## Shape all (bs,keypoints,3)
                # Denormalize D in predicted UVD and revert to global absolute coordinates
                preds = Normalize_depth(outputs, cubesize, com, add_com=True)
                # Denormalize D in "object detected" 2D coordinates UVD and revert to global absolute coordinates
                gt_crop = Normalize_depth(gt2Dcrop, cubesize, com, add_com=True)

                GT_crop.append(gt_crop)
                GT_UVD_orig.append(gt2Dorignal)
                GT_3D_orig.append(gt3Dorignal)
                GT_matrix.append(M_inv)
                estimation_cropped.append(preds)

                ## Calculate test loss
                inputs, gt_uvd, com, cubesize, joint_mask, visible_mask = \
                    data[0].to(device), data[1].to(device), data[4].to(device), \
                    data[6].to(device), data[7].to(device), data[8].to(device)
                test_loss, _ = ForwardPassFunc(model, inputs, gt_uvd, lossFunction, cubesize, com, joint_mask,
                                                  visible_mask)
                total_loss += test_loss
                count += 1

            print(f"Test Loss : {total_loss / count:.3f}")
            writer.add_scalar("test_loss", total_loss / count, epoch + 1)  # Test loss

            ## Stack along first dimension by default
            GT_crop = torch.cat(GT_crop).cpu()  # (B,K,joint_dim)
            GT_UVD_orig = torch.cat(GT_UVD_orig).cpu()  # (B,K,joint_dim)
            GT_3D_orig = torch.cat(GT_3D_orig).cpu()  # (B,K,joint_dim)
            GT_matrix = torch.cat(GT_matrix).cpu()  # (B,3,3)
            estimation_cropped = torch.cat(estimation_cropped).cpu()  # (B,K,joint_dim)

            ## Evaluation class
            # Parameters 1. predicted 2D keypoints uvd (cropped) 2. ground truth 2D keypoints uvd (cropped)
            Evaluator = HandposeEvaluation(estimation_cropped, GT_crop)

            output_message = ""

            # Cropped UVD
            if args.print_detail_crop:
                res = Evaluator.getErrorPerDimension(printOut=False)
                output_message = output_message + "\nUVD_Cropped:\n" + res + "\n###############################\n"

                output_message = output_message + f"\nThe error UVD in the cropped version={Evaluator.getMeanError():.3f}\n" + "###############################\n"

            # Original UVD: revert to original UVD
            prediction_UVDorig = CropToOriginal(estimation_cropped, GT_matrix.float())
            del estimation_cropped, GT_matrix
            Evaluator.update(prediction_UVDorig, GT_UVD_orig)
            # prediction_ArrayToFile(prediction_UVDorig.numpy(),"me.txt");

            if args.print_detail_uvd:
                res = Evaluator.getErrorPerDimension(printOut=False)
                output_message = output_message + "\nUVD_Original:\n" + res + "\n###############################\n"
                output_message = output_message + f"\nThe error in Original UVD ={Evaluator.getMeanError():.3f}\n" + "###############################\n"

            # 3D XYZ
            estimation_xyz = test_set.convert_uvd_to_xyz_tensor(prediction_UVDorig)
            Evaluator.update(estimation_xyz, GT_3D_orig)

            if args.print_detail_xyz:
                res = Evaluator.getErrorPerDimension(printOut=False)
                output_message = output_message + "\n3D results:\n" + res + "\n###############################\n"

            final_3Derror = Evaluator.getMeanError()

            output_message = output_message + f"Final 3D error results: {final_3Derror:.3f}mm\n" + 100 * "="

            ## Write results to log each round
            print(output_message)

            # Save model if performance is best
            if  final_3Derror < best_3Derror:
                best_3Derror = final_3Derror
                best_epoch = epoch+1
                model_name = f"Best_E{epoch + 1}_{final_3Derror:.3f}.pt"
                make_checkpoint(model_name, model, optimizer, scheduler, args)

            else:
                model_name=f"savedModel_E{epoch+1}_{final_3Derror:.3f}.pt"
                make_checkpoint(model_name, model, optimizer, scheduler, args)

            writer.add_scalar("Final 3D error results(mm)",final_3Derror , epoch + 1)  # Record average 3D error in mm for this round

    writer.close()
    print(f"The best epoch:{best_epoch},3D error:{best_3Derror}mm")
    return
