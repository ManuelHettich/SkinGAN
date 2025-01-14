# SkinGAN Project

`src` folder includes all the project codes. 

All the needed packages are listed in `requirements.txt`.

`train.py` is used for training the GAN. 
`inference.py` is used for modify a non-acne image with a pretrained GAN model.

`data` folder is the location for training data.

The ACNE04 images can be downloaded from [here](https://drive.google.com/drive/folders/18yJcHXhzOv7H89t-Lda6phheAicLqMuZ?usp=sharing).
Please download the Classification.tar and unzip it in the data folder so that the images can be found in data/Classification/JPEGImages.

The extracted facial landmarks are stored in `data/Results`.

`ckpt` folder contains a checkpoint of a pretrained Generator.

----------------------------------------------------
To train the model, you can use command: 

```
 python -m src.train --gpu 0 --exp train \
            --modify_loss_weight 1.0 --log_every 20 --epoch 1000 --batch_size 128  \
            --Dprestep 20 --Gprestep 0 --Gstep 2
```

To modify a no-acne image, you can use command:

```
python -m src.inference \
        --image_path 'data/example/levle0_2.jpg' \
        --landmark_path 'data/example/levle0_2.pkl' \
        --save_dir 'generate' \
        --Gckpt 'ckpt/Gen_5600.pth' \
        --num_patch 5
```
It will randomly choose a patch, add acnes to the patch and save the modified full face image and a patch comparison figure to `save_dir`. With `--num_patch X`, it will randomly sample X patches and generate X modifications.

To modify a single non-acne image with multiple patches, you can use this command:

```
python -m src.inference_combined \
        --image_name 'levle0_2' \
        --save_dir 'generate' \
        --Gckpt 'ckpt/Gen_5600.pth' \
        --num_patch 15
```

To modify a random selection of non-acne images (level 0) with multiple patches each, you can use this command:

```
python -m src.inference_dataset \
        --save_dir 'generate/dataset1' \
        --Gckpt 'ckpt/Gen_5600.pth' \
        --num_patch 10 \
        --num_images 10
```

## Cite as:
Lu, Z & Krishnamurthy, S (2021). 
SkinGAN: Medical image Synthetic data generation using Generative methods

