# ArcFace

This is the implementation of ArcFace by Python/Tensorflow. The original paper is [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)

## Network structure

ArcFace adds a penalty to the normal ResNet50/100 output and separate the distance of each class. This network is used to face recognition.

![Network structure](https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/arcface_network.png)

In use of ArcFace, we use 512-dims vectors as embeddigs.
![Use case](https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/arcface_example.png)

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
                  --backbone=ResNet50 \
                  --optimizer=SGD \
                  --epochs=100 \
                  --lr=0.001 \
                  --enhance=65 \
                  --penalty=0.5 \
                  --decay=1e-4 \
                  --dropout=0.5 \
                  --batch_size= 10 \
                  --save_path=/path/to/your/saving/folder \
```

The All commandline arguments have default value.

- n_classes: 10
- train_data: ./
- validation_data: ./
- backbone: ResNet50 (You can also use VGG16)
- optimizer: SGD (You can also use Adam)
- epochs: 100
- lr: 0.1
- enhance: 65
- penalty: 0.5
- batch_size: 10
- save_path: ./params

If you want to use pretrained parameters for training you can use '--use_param_folder' argument. This argument designate the directory where h5 file in.

Under construction
