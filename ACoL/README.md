# Adversarial Complementary Learning (ACoL)

This is the implementation of ACoL by Python with Tensorflow. The original paper is [Adversarial Complementary Learning for Weakly Supervised Object Localization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.pdf)

## Network structure

ACoL has two branches after VGG16 backbone to detect objects. To train this model, we use classifiers.

![Network structure](https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/ACoL_network.png)

The segmentation result introduced in the paper is following,

![Segmentation results](https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/ACoL_segmentation.png)

## How to train

Prepare your dataset like

```
.
├── train
│   ├── butterfly
│   ├── cat
│   ├── chicken
│   ├── cow
│   ├── dog
│   ├── elephant
│   ├── hourse
│   ├── sheep
│   ├── spider
│   └── squirrel
└── validation
    ├── butterfly
    ├── cat
    ├── chicken
    ├── cow
    ├── dog
    ├── elephant
    ├── hourse
    ├── sheep
    ├── spider
    └── squirrel
```

This directory structure is for classifying ten classes. Each directory has jpeg images.

```
$ python train.py --n_classes=10 \
                  --train_data=/path/to/your/dataset \
                  --validation_data=/path/to/your/dataset \
                  --epoch=100 \
                  --lr=0.001 \
                  --batch_size=20 \
                  --save_params=/path/to/your/saving/folder \
                  --threshold=0.85 \
```

The All commandline arguments have default value.

- n_classes: 10
- train_data: ./
- validation_data: ./
- epoch: 100
- lr: 0.001 ('--learning_rate' is also allowed)
- batch_size: 20
- save_params: ./
- threshold: 0.85

If you want to use pretrained parameters for training you can use '--use_param' argument. This argument designate the directory where h5 file in.

Under construction
