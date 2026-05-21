# KAD-Net: Kinematics-Aware Decoupled Learning for Robust 3D Hand Pose Estimation from a Single Depth Image

This repository contains the official implementation of the paper

## Preparing the Dataset

1. **Download the ICVL dataset** from the [ICVL Hand Posture Dataset](http://www.iis.ee.ic.ac.uk/~dtang/hand.html).

2. **Download the test set**: Download the file `test.pickle` from [here](http://www.iis.ee.ic.ac.uk/~dtang/hand.html).

3. **Prepare the training set**: extract the ICVL training set. **Run the preprocessing script** `python prepareICVL_train.py <ICVLpath>` to generate train.pickle. Here, `<ICVLpath>` represents the root path of the extracted ICVL training set (the folder containing `Depth/` and `labels.txt`).

5. **Organize the dataset folder**: Place both `test.pickle` and `train.pickle` in one folder. This folder will serve as the ICVL dataset folder used for training and evaluation.

    ```
    ICVL/
    ├── test.pickle
    └── train.pickle
    ```

## Configuration

Before running experiments, set the ICVL`datasetpath` value in the corresponding `.yaml` file located in the `configs/` folder. This value should point to the dataset folder (the one containing `train.pickle` and `test.pickle`).  in `configs/icvl.yaml`:

```yaml
datasetpath: "/path/to/ICVL"
```

## Training and Evaluation

Training：execute `bash train_eval_ICVL.bash`

Evaluation：execute `python eval.py`The result is in the file "result.txt".

we provided the pre-trained models ('./pretrained_model/icvl/best_model.pt') for ICVL



