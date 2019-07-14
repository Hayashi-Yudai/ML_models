# Adversarial Complementary Learning (ACoL)

This is the implementation of ACoL by Python with Tensorflow. The original paper is [Adversarial Complementary Learning for Weakly Supervised Object Localization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.pdf)

## Network structure

ACoL has two branches after VGG16 backbone to detect objects. To train this model, we use classifiers.

![Network structure](https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/ACoL_network.png)

The segmentation result introduced in the paper is following,

![Segmentation results](https://github.com/Hayashi-Yudai/ML_models/blob/master/Images/ACoL_segmentation.png)

Under construction
