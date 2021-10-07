# ImageNet training
This repository contains modification of pytorch example script[^1] for ImageNet training.

Modified script is contained in the file main_stratified.py.

Modified script extracts an extra validation set from the training set before training and in each epoch it saves trained network outputs on training set and both validation sets.

## Usage

Additional parameters of the modified script are:
- validation-size 	- size of subset of train data set aside for extra validation,
- output-folder 	- path to a folder in which training outputs will be stored,
- existing-val-split 	- path to a folder with files val_idx.npy and train_idx.npy specifying training/validation split of training set, None if there is no existing split,
- from-ptm 		- whether to load the model from library pretrainedmodels.

## Output

Script produces following output into the output folder:
- train_idx.npy			-indexes into the original train set used for training
- val_idx.npy			-indexes into the original train set uset for validation 2
- val2_output_{epoch}.npy	-output for validation set 2 in epoch
- val2_target.npy		-targets for validation set 2
- val_output_{epoch}.npy	-output for validation set in epoch
- val_target.npy		-targets for validation set
- train_output_{epoch}.npy	-output for training set in epoch
- train_target_{epoch}.npy	-targets for training set in epoch														
- valid_summary.txt		- {epoch},{losses.avg:.4e},{top1.avg:.3f},{top5.avg:.3f},\t{losses.avg:.4e},{top1.avg:.3f},{top5.avg:.3f}\n - first three values correspond to training data, second three to validation data
- checkpoint.pth.tar			-checkpoint rewritten every epoch
- model_best.pth.tar			-checkpoint with best top1 validation accuracy
	
	
[^1]: https://github.com/pytorch/examples/tree/master/imagenet
