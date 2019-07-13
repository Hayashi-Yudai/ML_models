# ML Models

This repository implements various machine learning models with Python/Tensorflow. I treat mainly "Image processing" in it.

|  Model  |              Paper               |
| :-----: | :------------------------------: |
|  U-Net  | https://arxiv.org/abs/1505.04597 |
|  ACoL   | https://arxiv.org/abs/1804.06962 |
| Arcface | https://arxiv.org/abs/1801.07698 |

You can use these models for training or validation.

## Requirements

- Python 3.6>=
- Tensorflow 1.12.0>=
- PIL
- Imgaug
- Numpy
- Scipy
- Matplotlib

To install these libraries, execute following command.

bash/Command prompt

```
$ pip install -r requirements.txt
```

## Usage

How to use each model is written in README in the each model. Basically you can training with

```
$ python train.py $(options)
```

## Licence

ML models is licenced under the MIT licence.

(C) Copyright 2019, Yudai Hayashi
