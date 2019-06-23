# energy_constrained_compression
Code for paper "ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model" (https://arxiv.org/abs/1812.01803)
```
@inproceedings{yang2018ecc,
  title={ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model},
  author={Yang, Haichuan and Zhu, Yuhao and Liu, Ji},
  booktitle={CVPR},
  year={2019}
}
```
## Prerequisites


```
Python (3.6)
PyTorch 1.0
```

To use the ImageNet dataset, download the dataset and move validation images to labeled subfolders (e.g., using https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training and testing


### example
To run the training with (TX2) energy constraint on MobileNet (with multiple GPUs),

```
python admm_prox_train.py --net mobilenet-imagenet --dataset imagenet --datadir ./ILSVRC_CLS/ --batch_size 128 --num_workers 8 --plr 1e-5 --l2wd 1e-4 --pslr 3.0 --psgrad_clip 1.0 --budget 0.0077 --zinit 0.0 --rho_z 10.0 --rho_y 10.0 --distill 0.5 --val_batch_size 512 --energymodel ./energymodel_mbnet_tx2.pt --logdir log/default/ --padam --psgrad_mask --mgpu --save_interval 1

```

### usage

#### Collect empirical measurements of energy consumption
```
usage: energy_tr_gen.py [-h] [--net NET] [--num NUM] [--gpuid GPUID]
                        [--num_classes NUM_CLASSES] [--test_num TEST_NUM]
                        [--conv] [--cpu] [--outfile OUTFILE]

Generate Energy Cost Data

optional arguments:
  -h, --help            show this help message and exit
  --net NET             network architecture
  --num NUM             number of samples to generate
  --gpuid GPUID         gpuid
  --num_classes NUM_CLASSES
                        number of classes
  --test_num TEST_NUM   number of repeated trails
  --conv                only use conv layers
  --cpu                 use cpu
  --outfile OUTFILE     the output file of generated data
```


#### Fit the energy estimation model
```
usage: energy_estimator.py [-h] [--infile INFILE] [--outfile OUTFILE]
                           [--net NET] [--preprocess PREPROCESS]
                           [--batch_size BATCH_SIZE] [--seed SEED]
                           [--epochs EPOCHS] [--wd WD] [--errhist ERRHIST]
                           [--pinv]

Energy Estimator Training

optional arguments:
  -h, --help            show this help message and exit
  --infile INFILE       the input file of training data
  --outfile OUTFILE     the output file of trained model
  --net NET             network architecture
  --preprocess PREPROCESS
                        preprocessor method
  --batch_size BATCH_SIZE
                        input batch size for training
  --seed SEED           random seed (default: 117)
  --epochs EPOCHS       number of epochs to train
  --wd WD               weight decay
  --errhist ERRHIST     the output of error history
  --pinv                use pseudo inverse to solve (only for bi-linear model)
```


#### Energy-constrained training
```
usage: admm_prox_train.py [-h] [--net NET] [--energymodel ENERGYMODEL]
                          [--dataset DATASET] [--datadir DATADIR]
                          [--batch_size BATCH_SIZE]
                          [--val_batch_size VAL_BATCH_SIZE]
                          [--num_workers NUM_WORKERS] [--epochs EPOCHS]
                          [--plr PLR] [--padam] [--padam_beta PADAM_BETA]
                          [--pslr PSLR] [--psgrad_mask]
                          [--psgrad_clip PSGRAD_CLIP] [--l2wd L2WD]
                          [--momentum MOMENTUM] [--zinit ZINIT]
                          [--yinit YINIT] [--lr_decay LR_DECAY]
                          [--s_int S_INT] [--randinit] [--pretrain PRETRAIN]
                          [--seed SEED] [--log_interval LOG_INTERVAL]
                          [--test_interval TEST_INTERVAL]
                          [--save_interval SAVE_INTERVAL] [--logdir LOGDIR]
                          [--distill DISTILL] [--rho_z RHO_Z] [--rho_y RHO_Y]
                          [--budget BUDGET] [--dadam] [--mgpu] [--slb SLB]
                          [--eval]

Model-Free Energy Constrained Training

optional arguments:
  -h, --help            show this help message and exit
  --net NET             network arch
  --energymodel ENERGYMODEL
                        energy prediction model file
  --dataset DATASET     dataset used in the experiment
  --datadir DATADIR     dataset dir in this machine
  --batch_size BATCH_SIZE
                        batch size for training
  --val_batch_size VAL_BATCH_SIZE
                        batch size for evaluation
  --num_workers NUM_WORKERS
                        number of workers for training
  --epochs EPOCHS       number of epochs to train
  --plr PLR             primal learning rate
  --padam               use adam for primal net update
  --padam_beta PADAM_BETA
                        betas of adam for primal net update
  --pslr PSLR           primal learning rate for sparsity
  --psgrad_mask         update s only when s.grad < 0
  --psgrad_clip PSGRAD_CLIP
                        clip s.grad to
  --l2wd L2WD           l2 weight decay
  --momentum MOMENTUM   primal momentum (if using sgd)
  --zinit ZINIT         initial dual variable z
  --yinit YINIT         initial dual variable y
  --lr_decay LR_DECAY   learning rate (default: 1)
  --s_int S_INT         how many batches for updating s
  --randinit            use random init
  --pretrain PRETRAIN   file to load pretrained model
  --seed SEED           random seed (default: 117)
  --log_interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
  --test_interval TEST_INTERVAL
                        how many epochs to wait before another test
  --save_interval SAVE_INTERVAL
                        how many epochs to wait before save a model
  --logdir LOGDIR       folder to save to the log
  --distill DISTILL     distill loss weight
  --rho_z RHO_Z         ADMM hyperparameter: rho for z
  --rho_y RHO_Y         ADMM hyperparameter: rho for y
  --budget BUDGET       energy budget
  --dadam               use adam for dual
  --mgpu                enable using multiple gpus
  --slb SLB             sparsity lower bound
  --eval                eval in the begining
```
