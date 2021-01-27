mainv4.py		- non-stratified split for calibration

main_stratified.py
Script for training imagenet architectures on altered training set.
Stratified validation set is cut from training set for ensemble weights training.

Script produces output into the output folder:
	train_idx.npy				-indexes into the original train set used for training
	val_idx.npy					-indexes into the original train set uset for validation 2
	val2_output_{epoch}.npy		-output for validation set 2 in epoch
	val2_target.npy				-targets for validation set 2
	val_output_{epoch}.npy		-output for validation set in epoch
	val_target.npy				-targets for validation set
	train_output_{epoch}.npy	-output for training set in epoch
	train_target_{epoch}.npy	-targets for training set in epoch
	
												training data											validation data
	valid_summary.txt			- {epoch},{losses.avg:.4e},{top1.avg:.3f},{top5.avg:.3f},\t{losses.avg:.4e},{top1.avg:.3f},{top5.avg:.3f}\n
	
	checkpoint.pth.tar			-checkpoint rewritten every epoch
	model_best.pth.tar			-checkpoint with best top1 validation accuracy