# U-Net

UNet is the semantic segmentation network proposed by O. Ronneberger, P. Fischer, and T. Brox in the [arXiv paper](https://arxiv.org/abs/1505.04597). This repository is the implementation of this model by the Python with the Tensorflow.

## Network structure

![Network structure](https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/UNet_network.png)

![Example](https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/UNet_segmentation.png)

## How to train

You can use some commandline arguments.
```bash
python train.py --train_data=path/to/your/train/data \
                --validation_data=path/to/your/validation/data \
                --learning_rate=0.001 \
                --batch_size=5 \
                --epoch=100 \
                --n_classes=2 \
                --l2=0.05
```

All arguments except for train_data and validation_data above are default value. As data for training, I provide some cat images (preparing teacher data for semantic segmentation is the hard work).

You can also use pipenv command for training.

```bash
pipenv run python trian.py (options)
```

## Training results
I executed the training for cat images as test with default commandline arguments. I show some of the images segmented with trained parameters.

<img src="https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/Unet_cat_1.png">
<img src="https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/Unet_cat_2.png">
