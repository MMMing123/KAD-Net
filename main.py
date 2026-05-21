import random

import torch
import numpy as np
import builtins
from dataloader import DATASET_NUM_JOINTS
import torch.optim as optim
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import logging


from config import *
from utils.utils import *
from engine import Train_Test


# Fix random seeds for reproducibility
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def main(args):
    # Create checkpoint directory
    if os.path.exists(args.checkpoints_dir):
        print("checkpoint dir already exists")
    else:
        os.mkdir(args.checkpoints_dir)
        os.mkdir(os.path.join(args.checkpoints_dir, "checkpoints"))
        print("checkpoint dir created")
    main_worker(args)
      

def main_worker(args):
    seed = args.main_seed
    seed_torch(seed)

    ########################## Model ##########################
    model= model_builder(args.model_name,num_joints=DATASET_NUM_JOINTS[args.dataset], args = args)
    model = model.cuda()

    default_cuda_id = "cuda:{}".format(args.default_cuda_id)
    device = torch.device(default_cuda_id)

    if args.use_logger:
            print("Logger will be used!")
            logger = getLogger(save_path = None, name = "Main", level = "INFO")
            builtins.print =  logger.info
    print("\n"+"##"*15 + "\n" + str(args) + "\n" + "##"*15 )


    ########################## Dataset and Optimizer ##########################
    data_loaders = {}

    labled_train = DATA_Getters(args)

    generator = torch.Generator()
    generator.manual_seed(seed)

    data_loaders["trainloader_labeled"] = torch.utils.data.DataLoader( labled_train, batch_size=args.batch_size,
               shuffle= True, num_workers=args.num_workers, pin_memory=True,drop_last=True,generator=generator)

    data_loaders["trainloader_unlabeled"] = None

    optimizer = get_optimizer(args.optimizer, model, args)
    scheduler = get_scheduler(optimizer, args)
    lossFunction = get_lossFunction(args.LossFunction)

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print("fp16_scaler being used!")

    if args.model_path is not None:
        load_checkpoint(model, args , optimizer, scheduler, device)


    print(f"Model to be trained: {args.model_name}")
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"############################## STATR_TRAIN_TEST ########################################")

    ########################## Main Loop ##########################
    Train_Test(model, data_loaders, args,lossFunction,optimizer,device,scheduler, fp16_scaler)

    print('Finished Training')

####################################


# Execution starts here when launched from bash
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    over_write_args_from_file(args, args.config_file)
    main(args)
    
    
