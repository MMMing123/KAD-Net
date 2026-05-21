import math
import cv2
import torch
from torch.utils.data.dataset import Dataset
# General
from PIL import Image
import numpy as np
import os.path
import scipy.io
import copy
import pickle
import struct
from scipy import stats, ndimage

# Training dataset sizes
DATASET_LENGTHS={}
DATASET_LENGTHS["nyu"] = 72757
DATASET_LENGTHS["icvl"] = 22067
DATASET_LENGTHS["msra"] = 76375 - 8500

# Number of joints in each dataset
DATASET_NUM_JOINTS = {}
DATASET_NUM_JOINTS["nyu"]=14
DATASET_NUM_JOINTS["icvl"]=16
DATASET_NUM_JOINTS["msra"]=21

# Parent class for all datasets, inherits from PyTorch Dataset
class HandPoseDataset(Dataset):
    def __init__(self, 
                basepath="",train=True,cropSize=(128,128),doJitterRotation=False,doAddWhiteNoise=False,rotationAngleRange=[-45.0, 45.0],
                 comJitter=False,RandDotPercentage=0,  sigmaNoise=2,cropSize3D=[280,280,280],
                 do_norm_zero_one=False,random_seed=21,drop_joint_num=0,center_refined=False,horizontal_flip=0, scale_aug=0): 

        # Depth value range captured by the depth camera; values outside this range are set to 0
        self.min_depth_cam = 50.
        self.max_depth_cam = 1500.

        # do_norm_zero_one: whether to normalize features to [0,1]; otherwise to [-1,1]
        self.do_norm_zero_one=do_norm_zero_one

        # Bool, whether to perform random rotation; set to 1 in the config file
        self.doJitterRotation=doJitterRotation

        # Random rotation angle range, [-45,45]; added separately in base_parameters
        self.rotationAngleRange=rotationAngleRange

        # Base path of the data folder
        self.basepath = basepath

        # Selected joint indices (14 selected from 36 for training and evaluation), default for NYU
        self.restrictedJointsEval = nyuRestrictedJointsEval

        # 3D crop size? [250,250,250]
        self.cropSize3D = cropSize3D

        # Parameter for randomly dropped pixels, i.e. alpha, PixDropOut
        self.RandDotPercentage = RandDotPercentage

        # Number of joints dropped per sample for data augmentation; set to 0 in config
        self.drop_joint_num=drop_joint_num

        # Bool, whether to use horizontal flip; set to 0 in config
        self.horizontal_flip=horizontal_flip

        # For comparisons check results with adapted cube size
        
        self.testseq2_start_id = 2441
        self.cropSize = cropSize
        self.doAddWhiteNoise = doAddWhiteNoise
        self.sigmaNoise = sigmaNoise
        self.doNormZeroOne = do_norm_zero_one  # [-1,1] or [0,1]
        self.center_refined=center_refined
        self.scale_aug=scale_aug
        self.randomAngle=0
        self.randomScale=1
        self.randomComJitter=0 * np.clip(np.random.randn(3)*6,-6,6)
        self.comJitter = comJitter
        
        self.doTrain = train
        self.seqName = ""
        if self.doTrain:
            self.seqName = "train"
        else:
            self.seqName = "test"

        self.numSamples=self.numSamples


        self.indeces = [i for i in range(self.numSamples)]

        if drop_joint_num>0:
            print(f"{drop_joint_num} out of {self.num_joints} joints will be dropped for each frame")
            joint_list=[i for i in range(self.num_joints)]
            np.random.seed(random_seed+10)
            self.dropped_joints=[np.random.choice(joint_list,drop_joint_num,replace=False) for j in range(self.numSamples)]

    def __len__(self):
        return self.numSamples

    def LoadSample(self,ind,com=None):
        """
        This function should be overwritten in subclasses
        """

        raise NotImplementedError()

    def convert_uvd_to_xyz_tensor(self,uvd):
        """
        This function should be overwritten in subclasses
        """

        raise NotImplementedError()

    def convert_xyz_to_uvd_tensor(self,xyz):
        """
        This function should be overwritten in subclasses
        """

        raise NotImplementedError()

    def pointImgTo3D(self,sample):
        """
        This function should be overwritten in subclasses
        """

        raise NotImplementedError()

    def __getitem__(self, index):
        valIndex = self.get_validIndex(index)

        # Redefine center points using file
        if self.center_refined: # center_refined defaults to 1
            com=self.center_refined_uvd[valIndex]
        else:
            com=None

        # Load different dictionary data based on dataset
        data = self.LoadSample(valIndex,com) # NOTE: depth map has been "object detected", keypoints have been adjusted accordingly, but no data processing: normalization

        # I. Normalization: default normalize to [-1,1]
        # Returns normalized depth map and 3D world coordinates (x,y,z all normalized)
        if self.doNormZeroOne:
            img, target = normalizeZeroOne(data)
        else:
            img, target = normalizeMinusOneOne(data)

        self.sample_loaded = data
        com = torch.from_numpy(data["com3D"])

        # Get normalized UVD coordinates, normalize D in UVD to [-1,1]: D coordinate minus center depth and scaled by cubesize
        gt2Dcrop = torch.from_numpy((data["gt2Dcrop"]-np.array([0,0,com[2]]))/np.array([1,1.,data['cubesize'][2] / 2.]))


        # II. Scale depth map: scale depth map based on center point (same size before and after scaling), and corresponding values, uvd also needs to be scaled
        if self.scale_aug and np.random.rand()<0.7: # remember in this case, M will not result in the original UVD
            scale = 0.8 + np.random.random() * 0.4 
            img,gt2Dcrop=scale_depth(img,gt2Dcrop,scale)
            self.randomScale=scale
        else:
            self.randomScale=1

        img = np.expand_dims(img, axis=0) # Expand dimension, convert to Tensor
        img=torch.from_numpy(img)

        # III. Pixel dropout
        # RandDotPercentage: alpha for pixel dropout
        if self.RandDotPercentage>0 and np.random.rand()<0.7:
            p=np.random.rand()*self.RandDotPercentage
            img=PixDropout(img,background_value=1,P=p,V=1)


        # IV. Calculate camera matrix M and inverse matrix M_inv, convert related data to tensors
        M_=torch.from_numpy(data["M"])
        M_=torch.cat( [ torch.cat([ M_[:,:2],torch.zeros(3,1),M_[:,2][...,None]],dim=-1), torch.zeros(1,4)]);M_[2,2]=1;M_[2,3]=0;
        M=M_.float()
        M_inv=torch.from_numpy(np.linalg.inv(data["M"]))
        M_=torch.cat( [ torch.cat([ M_inv[:,:2],torch.zeros(3,1),M_inv[:,2][...,None]],dim=-1), torch.zeros(1,4)]);M_[2,2]=1;M_[2,3]=0;
        M_inv=M_.float()

        cubesize = torch.from_numpy(np.array(data["cubesize"])).float()
        gt2Dorignal = torch.from_numpy(data["gt2Dorignal"]).float()
        gt3Dorignal = torch.from_numpy(data["gt3Dorignal"]).float()
        
        # V. Determine joint visibility
        # Check pixel values within a certain range around landmark coordinates in the image.
        # If the region is mostly valid (i.e., not background), the landmark is considered visible, mask position is True
        # gt2Dcrop shape: [num_joints, 3] ==> gt2Dcrop[:,:2] = [14,2]
        # visible_mask shape: [num_joints, 1] 1 for visible, 0 for invisible
        visible_mask=get_visible(img , gt2Dcrop[:,:2], cropSize=self.cropSize[0], background_value=1)  # joints that are visible within the image frame

        # joint_mask: boolean, represents whether joint mask is dropped, defaults to not dropped
        joint_mask=torch.ones(self.num_joints,1, dtype=torch.bool)
        if self.drop_joint_num>0:
            joint_mask[self.dropped_joints[valIndex]]=False

        # img[1,128,128] tensor;
        return img.float(), gt2Dcrop.float(), gt2Dorignal, gt3Dorignal, com.float(), M_inv, cubesize.float(), joint_mask.float(), visible_mask, M

    # Crop data based on hand center point (depth values also cropped according to cubic_size)
    # dsize = cropSize
    def cropArea3D(self, imgDepth, com, minRatioInside=0.75, size=(250, 250, 250), dsize=(128, 128)):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """
        RESIZE_BILINEAR = 0
        RESIZE_CV2_NN = 1
        RESIZE_CV2_LINEAR = 2
        CROP_BG_VALUE = 0.0
        resizeMethod = RESIZE_CV2_NN
        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = comToBounds(com.copy(), size, self.fx, self.fy)

        # Check if part within image is large enough; otherwise stop
        xstartin = max(xstart, 0)
        xendin = min(xend, imgDepth.shape[1])
        ystartin = max(ystart, 0)
        yendin = min(yend, imgDepth.shape[0])
        ratioInside = float((xendin - xstartin) * (yendin - ystartin)) / float((xend - xstart) * (yend - ystart))
        if (ratioInside < minRatioInside) \
                and ((com[0] < 0) \
                     or (com[0] >= imgDepth.shape[1]) \
                     or (com[1] < 0) or (com[1] >= imgDepth.shape[0])):
            print("Hand largely outside image (ratio (inside) = {})".format(ratioInside))
            raise UserWarning('Hand not inside image')

        # crop patch from source
        cropped = imgDepth[max(ystart, 0):min(yend, imgDepth.shape[0]),
                  max(xstart, 0):min(xend, imgDepth.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0), abs(yend) - min(yend, imgDepth.shape[0])),
                                   (abs(xstart) - max(xstart, 0), abs(xend) - min(xend, imgDepth.shape[1]))),
                         mode='constant', constant_values=int(CROP_BG_VALUE))
        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)
        # Backface is at 0, it is set later;
        # setting anything outside cube to same value now (was set to zstart earlier)
        cropped[msk1] = CROP_BG_VALUE
        cropped[msk2] = CROP_BG_VALUE

        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = np.asmatrix(np.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        # Compute size of image patch for isotropic scaling
        # where the larger side is the side length of the fixed size image patch (preserving aspect ratio)
        if wb > hb:
            sz = (dsize[0], int(round(hb * dsize[0] / float(wb))))
        else:
            sz = (int(round(wb * dsize[1] / float(hb))), dsize[1])

        # Compute scale factor from cropped ROI in image to fixed size image patch;
        # set up matrix with same scale in x and y (preserving aspect ratio)
        roi = cropped
        if roi.shape[0] > roi.shape[1]:  # Note, roi.shape is (y,x) and sz is (x,y)
            scale = np.asmatrix(np.eye(3, dtype=float) * sz[1] / float(roi.shape[0]))
        else:
            scale = np.asmatrix(np.eye(3, dtype=float) * sz[0] / float(roi.shape[1]))
        scale[2, 2] = 1

        # depth resize
        if resizeMethod == RESIZE_CV2_NN:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)
        elif resizeMethod == RESIZE_BILINEAR:
            rz = HandDetector.bilinearResize(cropped, sz, CROP_BG_VALUE)
        elif resizeMethod == RESIZE_CV2_LINEAR:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError("Unknown resize method!")

        # Sanity check
        # numValidPixels = np.sum(rz != CROP_BG_VALUE)
        # if (numValidPixels < 40) or (numValidPixels < (np.prod(dsize) * 0.01)):
        #     print("Too small number of foreground/hand pixels: {}/{} ({}))".format(
        #         numValidPixels, np.prod(dsize), dsize))
        #     raise UserWarning("No valid hand. Foreground region too small.")

        # Place the resized patch (with preserved aspect ratio)
        # in the center of a fixed size patch (padded with default background values)
        ret = np.ones(dsize, np.float32) * CROP_BG_VALUE  # use background as filler
        xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
        xend = int(xstart + rz.shape[1])
        ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        off = np.asmatrix(np.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, off * scale * trans, com

# Selected joint indices (14 selected from 36 for training and evaluation)
nyuRestrictedJointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]

class NYUHandPoseDataset(HandPoseDataset):
    def __init__(self, basepath="",train=True,cropSize=(128,128),doJitterRotation=False,rotationAngleRange=[-45.0, 45.0],
                 comJitter=False,RandDotPercentage=0,  sigmaNoise=1,cropSize3D=[280,280,280],camID=1,
                 do_norm_zero_one=False,random_seed=21,drop_joint_num=0,center_refined=False,horizontal_flip=0, scale_aug=0):

            self.fx, self.fy, self.ux , self.uy = (588.036865, 587.075073, 320, 240)
            self.camID=camID
            self.num_joints = len(nyuRestrictedJointsEval)

            if train:
                self.seqName = "train"
            else:
                self.seqName = "test"

            # Load labels
            labels = '{}/{}/joint_data.mat'.format(basepath, self.seqName)
            self.labelMat = scipy.io.loadmat(labels) # Main dictionary, contains uvd and xyz

            # Get number of samples from annotations (test: 8252; train: 72757)
            self.numSamples = self.labelMat['joint_xyz'][camID-1].shape[0]

            super(NYUHandPoseDataset, self).__init__(basepath=basepath,train=train,cropSize=cropSize,doJitterRotation=doJitterRotation,rotationAngleRange=rotationAngleRange,
                        comJitter=comJitter,RandDotPercentage=RandDotPercentage, sigmaNoise=sigmaNoise,cropSize3D=cropSize3D,
                        do_norm_zero_one=do_norm_zero_one,random_seed=random_seed,drop_joint_num=drop_joint_num,center_refined=center_refined,horizontal_flip=horizontal_flip, scale_aug=scale_aug)


            ## center_refined_uvd redefined center point shape[num,3]
            if center_refined:
                if self.seqName == "train":
                    print("Center refined being used")
                center_path = os.path.join(basepath,'center_{}_refined.txt'.format(self.seqName))
                refined=np.loadtxt(center_path)
                self.center_refined_uvd = np.array(refined)# self.convert_xyz_to_uvd_tensor(torch.tensor(refined).unsqueeze(1))  ).squeeze() #(B,1, 3) -> (B,3)
            if self.seqName == "train":
                print("NYU Dataset init done.")


        # Returns a dictionary containing various information
        # Three steps: 1. Rotation (changes dpt and uvd) 2. Object detection based on center point (returns dpt, M, center point) 3. Change 2D/3D keypoint coordinates
        # Parameters: index, com
    def LoadSample(self,ind,com=None):
        # Center point index, if not using external file center point, use original center point
        idComGT = 13

        # Load the dataset
        objdir = '{}/{}/'.format(self.basepath,self.seqName) # seqName = 'train/test'

        # labelMat label dictionary
        joints3D = self.labelMat['joint_xyz'][self.camID-1] # [train/test count, 36, 3]
        joints2D = self.labelMat['joint_uvd'][self.camID-1] # [train/test count, 36, 3]

        # Selected joint index list
        eval_idxs = nyuRestrictedJointsEval
        numJoints = len(eval_idxs)

        line = ind

        # Build single depth map filename
        prefix = "depth"
        dptFileName = '{0:s}/{1:s}_{2:1d}_{3:07d}.png'.format(objdir, prefix, self.camID, line+1)

        # Get depth map data (numpy) [480,640] height-width
        dpt = loadDepthMap(dptFileName)

        ## Filter required joints
        # gt2Dorignal[14,3] filtered uvd
        gt2Dorignal = np.zeros((numJoints, 3), np.float32) # gt2Dorignal[14,3]
        jt = 0
        for ii in range(joints2D.shape[1]): # joints2D [train/test count, 36, 3]
            if ii not in eval_idxs:
                continue
            gt2Dorignal[jt,0] = joints2D[line,ii,0]
            gt2Dorignal[jt,1] = joints2D[line,ii,1]
            gt2Dorignal[jt,2] = joints2D[line,ii,2]
            jt += 1
        # gt3Dorignal[14,3] filtered x,y,z
        gt3Dorignal = np.zeros((numJoints,3),np.float32)
        jt = 0
        for jj in range(joints3D.shape[1]):
            if jj not in eval_idxs:
                continue
            gt3Dorignal[jt,0] = joints3D[line,jj,0]
            gt3Dorignal[jt,1] = joints3D[line,jj,1]
            gt3Dorignal[jt,2] = joints3D[line,jj,2]
            jt += 1

        ## Add some random jitter to COM position (distinguishes training and testing)
        self.randomComJitter = np.clip(np.random.randn(3)*6,-self.comJitter,self.comJitter)
        comGT = (copy.deepcopy(gt2Dorignal[idComGT]) if com is None else com)  +  self.randomComJitter

        ## Rotate depth map and keypoint coordinates UVD
        if self.doJitterRotation: # doJitterRotation corresponds to RotAugment in config, defaults to 1, rotationAngleRange added in utils, value is [-45,45]
            rotation_angle_scale = np.random.randn() # Scalar following normal distribution 0,1 range [-3,3]
            rot = rotation_angle_scale * (self.rotationAngleRange[1] - self.rotationAngleRange[0]) + self.rotationAngleRange[0]
            self.randomAngle=rot # randomAngle range [-135, 135]
            dpt, gt2Dorignal = rotateImageAndGt(dpt, comGT, rot, gt2Dorignal,bgValue=10000)

        ## Crop: crop data based on hand center point, essentially still cropping, so next need to offset 2D/3D keypoints
        # doTrain Boolean representing training/test data; cropSize3D [288,288,288]
        cubesize = np.float32( np.array(self.cropSize3D) * (5/6. if (not self.doTrain and ind>=2440) else 1) )
        # Returns data dpt[128,128] comprehensive transformation matrix M[3,3] com[3]
        dpt, M, com = self.cropArea3D(imgDepth=dpt,com=comGT,minRatioInside=0.6 ,size=cubesize, dsize=self.cropSize)
        # Convert to 3D center point
        com3D = self.pointImgTo3D(com)

        ## 2D and 3D keypoint positions also need to offset (based on returned comprehensive matrix): 2D directly offset, 3D becomes relative to center point
        gt3Dcrop = gt3Dorignal - com3D     # normalize to com
        gt2Dcrop = np.zeros((gt2Dorignal.shape[0], 3), np.float32)
        for joint in range(gt2Dorignal.shape[0]):
            t=transformPoint2D(gt2Dorignal[joint], M)
            gt2Dcrop[joint, 0] = t[0]
            gt2Dcrop[joint, 1] = t[1]
            gt2Dcrop[joint, 2] = gt2Dorignal[joint, 2]

        ## Store related data in dictionary and return
        D={};D["M"]=M;D["com3D"]=com3D;D["cubesize"]=cubesize
        D["dpt"]=dpt.astype(np.float32);D["gt2Dorignal"]=gt2Dorignal;D["gt3Dorignal"]=gt3Dorignal
        D["gt2Dcrop"]=gt2Dcrop;D["gt3Dcrop"]=gt3Dcrop

        return D

    def convert_uvd_to_xyz_tensor(self, uvd ):
        # uvd is a tensor of  size(B,num_joints,3)
        xRes = 640;
        yRes = 480;
        xzFactor = 1.08836710; #xzFactor=640/coeffX
        yzFactor = 0.817612648;

        normalizedX = uvd[:,:,0] / xRes - 0.5;
        normalizedY = 0.5 - uvd[:,:,1] / yRes;

        xyz = torch.zeros(uvd.shape);
        xyz[:,:,2] = uvd[:,:,2];
        xyz[:,:,0] = normalizedX * xyz[:,:,2] * xzFactor;
        xyz[:,:,1] = normalizedY * xyz[:,:,2] * yzFactor;
        return xyz


    def convert_xyz_to_uvd_tensor(self, xyz):
        uvd = torch.zeros(xyz.shape);
        
        uvd[:,:,2] = xyz[:,:,2];

        uvd[:,:,0] = xyz[:,:,0]/xyz[:,:,2]*self.fx+self.ux
        
        uvd[:,:,1] = self.uy-xyz[:,:,1]/xyz[:,:,2]*self.fy
        return uvd
       

# Convert a uvd point to xyz
    def pointImgTo3D(self, sample):
        ret = np.zeros((3,), np.float32)
        # convert to metric using f, see Thomson et al.
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret

    # NYU all valid
    def get_validIndex(self, ind):

        return self.indeces[ind]


########## ICVL ##########################################

InvalidIndicies=[6635, 8001, 9990, 10292, 12378, 19770, 19863, 20531] + [7037, 8724, 11852, 19161, 21080 , 10463 , 12837  , 19864 , 20343 , 20532 ] # the second for the case where we have center_jitter


class ICVLHandPoseDataset(HandPoseDataset):
    def __init__(self, basepath="", train=True, cropSize=(128, 128), doJitterRotation=False, doAddWhiteNoise=False,
                 rotationAngleRange=[-45.0, 45.0],
                 comJitter=False, RandDotPercentage=0, sigmaNoise=1, cropSize3D=[250,250,250],
                 do_norm_zero_one=False, random_seed=21,
                 drop_joint_num=0, center_refined=False, horizontal_flip=0, scale_aug=0):

        self.fx, self.fy, self.ux, self.uy = (240.99, 240.96, 160, 120)  # (241.42, 241.42, 160., 120.)

        self.num_joints = 16

        self.seqName = ""
        if train:
            self.seqName = "train"
        else:
            self.seqName = "test"

        address = os.path.join(basepath, "%s.pickle" % (self.seqName))

        data = pickle.load(open(address, "rb"))
        self.data = data[0]

        self.numSamples = len(self.data)

        super(ICVLHandPoseDataset, self).__init__(basepath=basepath, train=train, cropSize=cropSize,
                                                  doJitterRotation=doJitterRotation, doAddWhiteNoise=doAddWhiteNoise,
                                                  rotationAngleRange=rotationAngleRange,
                                                  comJitter=comJitter, RandDotPercentage=RandDotPercentage,
                                                 sigmaNoise=sigmaNoise, cropSize3D=cropSize3D,
                                                  do_norm_zero_one=do_norm_zero_one, random_seed=random_seed,
                                                  drop_joint_num=drop_joint_num, center_refined=center_refined,
                                                  horizontal_flip=horizontal_flip, scale_aug=scale_aug)

        if center_refined:
            #print("Center refined being used")
            self.center_refined_uvd = data[1]

      #  print("ICVL Dataset init done.")


    ## Some datasets in ICVL are incorrectly marked, recorded as invalid
    def get_validIndex(self, ind):

        valIndex = self.indeces[ind]
        while valIndex in InvalidIndicies:
            valIndex = self.indeces[np.random.randint(0, self.numSamples)]

        return valIndex

    def LoadSample(self, ind, com=None):
        idComGT = 0
        # Load the dataset

        sample = self.data[ind]

        gt3Dorignal = sample[2]
        gt2Dorignal = sample[1]

        numJoints = gt2Dorignal.shape[0]

        dpt = sample[0]

        self.randomComJitter = np.clip(np.random.randn(3) * 6, -self.comJitter,
                                       self.comJitter)  # (1 if self.comJitter else 0)* np.clip(np.random.randn(3)*6,-6,6)
        comGT = (copy.deepcopy(gt2Dorignal[
                                   idComGT]) if com is None else com) + self.randomComJitter  # np.clip(np.concatenate((np.random.randn(2),np.array([0.])))*6,-6,6)

        # Add noise?
        if self.doAddWhiteNoise:
            img_white_noise_scale = np.random.randn(dpt.shape[0], dpt.shape[1])
            dpt = dpt + sigmaNoise * self.img_white_noise_scale

        if self.doJitterRotation:
            rotation_angle_scale = np.random.randn()
            rot = rotation_angle_scale * (self.rotationAngleRange[1] - self.rotationAngleRange[0]) + \
                  self.rotationAngleRange[0]
            self.randomAngle = rot
            dpt, gt2Dorignal = rotateImageAndGt(dpt, comGT, rot, gt2Dorignal, bgValue=10000)

        # Jitter scale (cube size)?
        cubesize = self.cropSize3D

        dpt, M, com = self.cropArea3D(imgDepth=dpt, com=comGT, minRatioInside=0.6, size=cubesize, dsize=self.cropSize)

        com3D = self.pointImgTo3D(com)
        gt3Dcrop = gt3Dorignal - com3D  # normalize to com
        gt2Dcrop = np.zeros((gt2Dorignal.shape[0], 3), np.float32)
        for joint in range(gt2Dorignal.shape[0]):
            t = transformPoint2D(gt2Dorignal[joint], M)
            gt2Dcrop[joint, 0] = t[0]
            gt2Dcrop[joint, 1] = t[1]
            gt2Dcrop[joint, 2] = gt2Dorignal[joint, 2]

        D = {};
        D["M"] = M;
        D["com3D"] = com3D;
        D["cubesize"] = cubesize
        D["dpt"] = dpt.astype(np.float32);
        D["gt2Dorignal"] = gt2Dorignal;
        D["filename"] = sample[-1]
        D["gt2Dcrop"] = gt2Dcrop;
        D["gt3Dorignal"] = gt3Dorignal;
        D["gt3Dcrop"] = gt3Dcrop;
        return D

    def convert_uvd_to_xyz_tensor(self, uvd):
        # uvd is a tensor of  size(B,num_joints,3)

        xyz = torch.zeros(uvd.shape);
        xyz[:, :, 2] = uvd[:, :, 2];
        xyz[:, :, 0] = (uvd[:, :, 0] - self.ux) * uvd[:, :, 2] / self.fx
        xyz[:, :, 1] = (uvd[:, :, 1] - self.uy) * uvd[:, :, 2] / self.fy
        return xyz

    def convert_xyz_to_uvd_tensor(self, xyz):
        # xyz is a tensor of  size(B,num_joints,3)
        uvd = torch.zeros(xyz.shape);

        uvd[:, :, 2] = xyz[:, :, 2];

        uvd[:, :, 0] = xyz[:, :, 0] / xyz[:, :, 2] * self.fx + self.ux

        uvd[:, :, 1] = xyz[:, :, 1] / xyz[:, :, 2] * self.fy + self.uy
        return uvd

    def pointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (sample[1] - self.uy) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret



########################################### MSRA ###################################
# Num_frame_per_each_subject = [ 8499, 8492, 8412, 8488, 8500, 8497, 8497, 8498, 8492]
class MSRAHandPoseDataset(HandPoseDataset):
    def __init__(self, basepath="",train=True,cropSize=(128,128),doJitterRotation=False,doAddWhiteNoise=False,rotationAngleRange=[-45.0, 45.0],
                 comJitter=False,RandDotPercentage=0,  sigmaNoise=1,cropSize3D=[280,280,280], do_norm_zero_one=False,random_seed=21,
                 drop_joint_num=0,center_refined=False,horizontal_flip=0, scale_aug=0, LeaveOut_subject=0 , use_default_cube=True):



        self.fx, self.fy, self.ux , self.uy = (241.42, 241.42, 160., 120.)

        self.num_joints = 21

        self.default_cubes = {'P0': [200, 200, 200],
                              'P1': [200, 200, 200],
                              'P2': [200, 200, 200],
                              'P3': [180, 180, 180],
                              'P4': [180, 180, 180],
                              'P5': [180, 180, 180],
                              'P6': [170, 170, 170],
                              'P7': [160, 160, 160],
                              'P8': [150, 150, 150]}


        self.address_gtUVD = os.path.join(basepath, "msra_test_groundtruth_label.txt")
        self.address_files = os.path.join(basepath, "msra_test_list.txt")
        self.use_default_cube = use_default_cube

        data = {}

        gt = open(self.address_gtUVD);gt.seek(0);
        gt_lines = gt.readlines()

        files = open(self.address_files);files.seek(0);
        file_lines = files.readlines()

        assert len(gt_lines) == len(file_lines)

        for i in range(len(file_lines)):
            part = gt_lines[i].split(' ')
            gt_uvd_original = np.zeros((self.num_joints, 3), np.float32)

            for joint in range(self.num_joints):
                for xyz in range(0, 3):
                    gt_uvd_original[joint, xyz] = part[joint*3+xyz]


            depth_address = os.path.join(basepath, file_lines[i][:-1])
            subject = file_lines[i].split("/")[0]

            data[subject] = data.get(subject,[]) + [ (gt_uvd_original, subject, depth_address) ]



        if train:
            self.seqName = "train"

            # for key,value in data.items():
            #     if key != f"P{LeaveOut_subject}":
            #         print(key,len(data[key]))
            #     else:
            #         print(key,len(data[key])," **")

            data = [data[key] for key in data.keys() if key != f"P{LeaveOut_subject}"]
            self.data = [item for sublist in data for item in sublist]

        else:
            self.seqName = "test"
            self.data=data[f"P{LeaveOut_subject}"]



        self.numSamples = len(self.data)



        super(MSRAHandPoseDataset, self).__init__(basepath=basepath,train=train,cropSize=cropSize,doJitterRotation=doJitterRotation,doAddWhiteNoise=doAddWhiteNoise,rotationAngleRange=rotationAngleRange,
                    comJitter=comJitter,RandDotPercentage=RandDotPercentage,  sigmaNoise=sigmaNoise,cropSize3D=cropSize3D,
                    do_norm_zero_one=do_norm_zero_one,random_seed=random_seed,drop_joint_num=drop_joint_num,center_refined=False,horizontal_flip=horizontal_flip, scale_aug=scale_aug)


        self.center_refined = center_refined

        if center_refined:
            if self.seqName == "train":
                print("Center refined being used")
            self.center_refined_uvd = [i for i in range(self.numSamples)]


        print("MSRA Dataset init done.")





    def LoadSample(self,ind,com=None):
        idComGT = 9
        # Load the dataset

        gt_uvd_original, subject, depth_address = self.data[ind]
        self.depth_address = depth_address

        gt3Dorignal = self.joints3DToImg(gt_uvd_original)
        gt2Dorignal = gt_uvd_original

        numJoints = gt2Dorignal.shape[0]


        dpt = self.loadDepthMap(depth_address).copy()
        original_dpt = dpt.copy()


        com = calculateCoM(dpt,minDepth=0,maxDepth=3000)
        self.com=com

        self.randomComJitter = np.clip(np.random.randn(3)*6,-self.comJitter,self.comJitter)# (1 if self.comJitter else 0)* np.clip(np.random.randn(3)*6,-6,6)
        comGT = (copy.deepcopy(gt2Dorignal[idComGT]) if com is None else com)  +  self.randomComJitter #np.clip(np.concatenate((np.random.randn(2),np.array([0.])))*6,-6,6)

        # Add noise?
        if self.doAddWhiteNoise:
            img_white_noise_scale = np.random.randn(dpt.shape[0], dpt.shape[1])
            dpt = dpt + sigmaNoise * self.img_white_noise_scale


        if self.doJitterRotation:
            rotation_angle_scale = np.random.randn()
            rot = rotation_angle_scale * (self.rotationAngleRange[1] - self.rotationAngleRange[0]) + self.rotationAngleRange[0]
            self.randomAngle=rot
            dpt, gt2Dorignal = rotateImageAndGt(dpt, comGT, rot, gt2Dorignal,  bgValue=10000)






        # Jitter scale (cube size)?
        cubesize = (self.default_cubes[subject] if self.use_default_cube else self.cropSize3D)


        dpt, M, com = self.cropArea3D(imgDepth=dpt,com=comGT,minRatioInside=0.6,size=cubesize, dsize=self.cropSize)

        com3D = self.pointImgTo3D(com)
        gt3Dcrop = gt3Dorignal - com3D     # normalize to com
        gt2Dcrop = np.zeros((gt2Dorignal.shape[0], 3), np.float32)
        for joint in range(gt2Dorignal.shape[0]):
            t=transformPoint2D(gt2Dorignal[joint], M)
            gt2Dcrop[joint, 0] = t[0]
            gt2Dcrop[joint, 1] = t[1]
            gt2Dcrop[joint, 2] = gt2Dorignal[joint, 2]

        D={};D["M"]=M;D["com3D"]=com3D;D["cubesize"]=cubesize
        D["dpt"]=dpt.astype(np.float32);D["gt2Dorignal"]=gt2Dorignal;D["filename"]=depth_address
        D["gt2Dcrop"]=gt2Dcrop;D["gt3Dorignal"]=gt3Dorignal;D["gt3Dcrop"]=gt3Dcrop;D["original_dpt"]=original_dpt;
        return D


    def convert_uvd_to_xyz_tensor( self, uvd ):
        # uvd is a tensor of  size(B,num_joints,3)

        xyz = torch.zeros(uvd.shape);
        xyz[:,:,2] = uvd[:,:,2];
        xyz[:,:,0] = (uvd[:,:,0]-self.ux)*uvd[:,:,2]/self.fx
        xyz[:,:,1] = (uvd[:,:,1]-self.uy)*uvd[:,:,2]/self.fy
        return xyz


    def convert_xyz_to_uvd_tensor(self, xyz):
        # xyz is a tensor of  size(B,num_joints,3)
        uvd = torch.zeros(xyz.shape);

        uvd[:,:,2] = xyz[:,:,2];

        uvd[:,:,0] = xyz[:,:,0]/xyz[:,:,2]*self.fx+self.ux

        uvd[:,:,1] = xyz[:,:,1]/xyz[:,:,2]*self.fy+self.uy
        return uvd



    def pointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret


    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.pointImgTo3D(sample[i])
        return ret


    def loadDepthMap(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """
        with open(filename, 'rb') as f:
            # first 6 uint define the full image
            width = struct.unpack('i', f.read(4))[0]
            height = struct.unpack('i', f.read(4))[0]
            left = struct.unpack('i', f.read(4))[0]
            top = struct.unpack('i', f.read(4))[0]
            right = struct.unpack('i', f.read(4))[0]
            bottom = struct.unpack('i', f.read(4))[0]
            patch = np.fromfile(f, dtype='float32', sep="")
            imgdata = np.zeros((height, width), dtype='float32')
            imgdata[top:bottom, left:right] = patch.reshape([bottom-top, right-left])


        return imgdata

    # Mask all valid
    def get_validIndex(self, ind):

        return self.indeces[ind]



########################################### Functions ############################################################

## comToBounds: get 2D boundaries using center point
# Method: calculate 3D boundaries of a region based on 2D com and cubic_size, perform spatial projection and offset, then back-project 3D info to 2D coordinates,
# Parameters size = cubic_size = [280,280,280]; com center point coordinates (uvd unless specified otherwise)
def comToBounds(com, size, fx, fy):
        """
        Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: xstart, xend, ystart, yend, zstart, zend
        """
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        # Principal point offset ux/uy affects both xstart and xend simultaneously, so not subtracting principal point offset doesn't affect their relative distance
        # 1. (com[0] * com[2] / fx - size[0] / 2. : find left boundary in 3D space starting from center point
        # 2. "/ com[2]*fx+0.5": transform 3D coordinates back to 2D image plane, add 0.5 for rounding up
        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx+0.5))
        xend = int(np.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2]*fx+0.5))
        ystart = int(np.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2]*fy+0.5))
        yend = int(np.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2]*fy+0.5))
        return xstart, xend, ystart, yend, zstart, zend

### Rotate UVD
# Parameters: original numpy depth map, center point, rotation angle (scalar), keypoint UVD coordinates
# Returns: rotated depth map, rotated UVD coordinates
def rotateImageAndGt(imgDepth, center,angle, gtUvd,bgValue=10000):
    """
    :param angle:   rotation angle
    :param center:   a tuple (x,y), which is going to be the center of the rotation 
        from image coordinates to 3D coordinates
        like transformations.pointsImgTo3D() (from the same file).
        (To enable specific projections like for the NYU dataset)
    """
    # Rotate image around given joint

    center =(center[0], center[1])
    rotationMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    sizeRotImg = (imgDepth.shape[1], imgDepth.shape[0])
    imgRotated = cv2.warpAffine(src=imgDepth, M=rotationMat, dsize=sizeRotImg, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=bgValue)
    
    # Rotate GT
    gtUvd_ = gtUvd.copy()
    gtUvdRotated = np.ones((gtUvd_.shape[0], 3), dtype=gtUvd.dtype)
    gtUvdRotated[:,0:2] = gtUvd_[:,0:2]
    gtUvRotated = np.dot(rotationMat, gtUvdRotated.T)
    gtUvdRotated[:,0:2] = gtUvRotated.T
    gtUvdRotated[:,2] = gtUvd_[:,2]
    
    return imgRotated, gtUvdRotated

## Determine visibility using some method
def get_visible(img , landmarks, cropSize=128, background_value=1,win_size=4):
    mask = torch.zeros(landmarks.shape[0],1, dtype=torch.bool)
    for j in range(landmarks.shape[0]):
        x = np.int32(np.round(landmarks[j,0]))
        y = np.int32(np.round(landmarks[j,1]))
        if x>=0 and x<cropSize and y>=0 and y<cropSize:
            left=max(0,x-win_size);right=min(x+win_size,cropSize)
            bottom=max(0,y-win_size);top=min(y+win_size,cropSize)
            window = img[0,bottom:top,left:right]
            if torch.sum(window)/window.numel() < background_value-1e-6 : 
                   mask[j,0]=True
  
    return torch.from_numpy( np.float32(mask) )

## Normalize depth image and cropped labels to [0,1]
# Input: dictionary
# Output: depth image and 3D world coordinates
def normalizeZeroOne(sample):
    imgD = np.asarray(sample["dpt"].copy(), 'float32')
    imgD[imgD == 0] = sample.com[2] + (sample['cubesize'][2] / 2.)
    imgD -= (sample["com3D"][2] - (sample['cubesize'][2] / 2.))
    imgD /= sample['cubesize'][2]
    
    target = np.clip(np.asarray(sample["gt3Dcrop"], dtype='float32') / sample['cubesize'][2], -0.5, 0.5) + 0.5
                
    return imgD, target
    
## Normalize depth image and cropped labels to [-1,1]
# Input: dictionary
# Output: depth image and 3D world coordinates
def normalizeMinusOneOne(sample):
    imgD = np.asarray(sample["dpt"].copy(), 'float32')
    imgD[imgD == 0] = sample["com3D"][2] + (sample['cubesize'][2] / 2.)
    # Image depth values become relative offsets to center point
    imgD -= sample["com3D"][2]
    # Scale to [-1,1]
    imgD /= (sample['cubesize'][2] / 2.)
    # Normalize 3D world coordinates x,y,z all to -1,1 (because gt3Dcrop is offset relative to center point)
    target = np.clip(np.asarray(sample["gt3Dcrop"], dtype='float32')/ (sample['cubesize'][2] / 2.), -1, 1)
    return imgD, target

def denormalize_depth(preds, sizes, coms, add_com=False):
    """
    Parameters:
        preds: normalized depth values with shape [bs, n]
        sizes: shape [bs, 3], dimensions for each sample (e.g., length, width, height)
        coms: shape [bs, 3], base point (e.g., center point) for each sample
        add_com: whether to add base point depth value to result
    Returns:
        Denormalized depth values with shape [bs, n]
    """
    # Extract depth dimension size (size[2])
    size_z = sizes[:, 2]  # Shape [bs]

    # Scale depth values: multiply by size_z / 2
    preds = preds * (size_z[:, None] / 2)  # Broadcast to [bs, n]

    # Optional: add base point depth value
    if add_com:
        com_z = coms[:, 2]  # Shape [bs]
        preds = preds + com_z[:, None]  # Broadcast to [bs, n]

    return preds
def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = np.asmatrix(M.reshape((3, 3))) * np.matrix([pt[0], pt[1], 1]).T
    return np.array([pt2[0] / pt2[2], pt2[1] / pt2[2]])


### Get numpy depth map data from filename
def loadDepthMap(filename):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    with open(filename) as f:
        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r,np.int32)
        g = np.asarray(g,np.int32)
        b = np.asarray(b,np.int32)
        dpt = np.bitwise_or(np.left_shift(g,8),b)
        imgdata = np.asarray(dpt,np.float32)

    return imgdata



def CropToOriginal(preds,matrices):
    # preds is a tensor of shape (B,K,3)
    # matrices is a tensor of shape (B,4,4), which is supposed to be the inverse matrix of each data sample
    v=torch.cat([preds,torch.ones(preds.shape[0],preds.shape[1],1).to(preds.device)],dim=-1)

    result=(v@matrices.transpose(-1,-2))
    return result[:,:,:3]

## Pixel dropout
def PixDropout(img,background_value,P,V=1):
  # image is a tensor of size (1,H,W)
  # background_value denotes the value using which the foreground pixels are computed
  # P is the percentage of foreground pixels that are assigned the value V

  # returns the same image with P percentage of its foregrounds pixels set to the value V

  img=img.clone()
  mask=abs(img[0]-background_value)>1e-6 # locate foreground pixels
  y,x=torch.where(mask) # get their pixel coordinates

  num_pix=np.int32(P*len(y))
  indecies_toSet=np.random.choice(len(x),size=num_pix,replace=False)
  img[0,y[indecies_toSet],x[indecies_toSet]]=V

  return img




## Horizontally flip depth map and uvd
def horizontal_flip_depth(img,uvd):
    #img np array of shape (d,d)
    #uvd tensor of np.arrayof shape (k,3)
    w=img.copy()
    for j in range(w.shape[1] // 2):
        tem = w[:, j].copy()
        w[:, j] = w[:, w.shape[1] - j - 1].copy()
        w[:, w.shape[1] - j - 1] = tem

    new_uvd=uvd.clone()
    new_uvd[:,0]=w.shape[1]-new_uvd[:,0]-1
    
    return w,new_uvd

## Scale depth map based on center point (same size before and after scaling), and corresponding values
# NOTE: uvd also needs to be scaled accordingly
def scale_depth(img,uvd,scale,background_value=1):
    # img is an np.array of size (d,d)
    # uvd is tensor or np.array of size (num_joint,dim)
    rows,cols,=img.shape
     
    # Create the transformation matrix
    center=((cols-1)/2.0,(rows-1)/2.0)

    # Get rotation matrix cv2.getRotationMatrix2D(center, angle, scale) (center point, angle, scale factor)
    # Use with cv2.warpAffine to scale image
    M = cv2.getRotationMatrix2D(center,0,scale)
    # Scale, use constant 1 to fill boundaries
    # NOTE: although size remains same before and after scaling, coordinate values will offset based on center point
    dst = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=1)

    ## Scale keypoints
    # Because rotation is based on center point, scale keypoints accordingly
    uvd[:,:2]=uvd[:,:2]-np.array(center)
    uvd=uvd*scale
    uvd[:,:2]=uvd[:,:2]+np.array(center)

    # Background value mask
    mask=abs(dst-background_value)<1e-6 #background pixels

    # Scale depth values
    dst=dst*scale
    dst[mask]=background_value

    return dst,uvd

def calculateCoM(dpt,minDepth=0,maxDepth=500):
    """
    Calculate the center of mass
    :param dpt: depth image
    :return: (x,y,z) center of mass
    """

    dc = dpt.copy()
    dc[dc < minDepth] = 0
    dc[dc > maxDepth] = 0
    cc = ndimage.measurements.center_of_mass(dc > 0)
    num = np.count_nonzero(dc)
    com = np.array((cc[1]*num, cc[0]*num, dc.sum()), np.float32)

    if num == 0:
        return np.array((0, 0, 0), np.float32)
    else:
        return com/num
