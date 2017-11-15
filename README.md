# Transfer Learning Lab with VGG, Inception and ResNet
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this lab, you will continue exploring transfer learning. You've already explored feature extraction with AlexNet and TensorFlow. Next, you will use Keras to explore feature extraction with the VGG, Inception and ResNet architectures. The models you will use were trained for days or weeks on the [ImageNet dataset](http://www.image-net.org/). Thus, the weights encapsulate higher-level features learned from training on thousands of classes.

We'll use two datasets in this lab:

1. German Traffic Sign Dataset
2. Cifar10

Unless you have a powerful GPU, running feature extraction on these models will take a significant amount of time. To make things we precomputed **bottleneck features** for each (network, dataset) pair, this will allow you experiment with feature extraction even on a modest CPU. You can think of bottleneck features as feature extraction but with caching.  Because the base network weights are frozen during feature extraction, the output for an image will always be the same. Thus, once the image has already been passed once through the network we can cache and reuse the output.

The files are encoded as such:

- {network}_{dataset}_bottleneck_features_train.p
- {network}_{dataset}_bottleneck_features_validation.p

network can be one of 'vgg', 'inception', or 'resnet'

dataset can be on of 'cifar10' or 'traffic'

How will the pretrained model perform on the new datasets?

Result
---

```
python feature_extraction.py --training_file vgg_cifar10_100_bottleneck_features_train.p --validation_file vgg_cifar10_bottleneck_features_validation.p

Epoch 50/50
1000/1000 [==============================] - 0s 188us/step - loss: 0.2763 - acc: 0.9400 - val_loss: 0.9125 - val_acc: 0.7077
```


```
python feature_extraction.py --training_file inception_cifar10_100_bottleneck_features_train.p --validation_file inception_cifar10_bottleneck_features_validation.p

Epoch 50/50
1000/1000 [==============================] - 0s 313us/step - loss: 0.0897 - acc: 1.0000 - val_loss: 1.0569 - val_acc: 0.6516

```

```
python feature_extraction.py --training_file resnet_cifar10_100_bottleneck_features_train.p --validation_file resnet_cifar10_bottleneck_features_validaion.p

Epoch 50/50
1000/1000 [==============================] - 0s 344us/step - loss: 0.0738 - acc: 1.0000 - val_loss: 0.8134 - val_acc: 0.7273

```

```
python feature_extraction.py --training_file vgg_traffic_100_bottleneck_features_train.p --validation_file vgg_traffic_bottleneck_features_validation.p

Epoch 50/50
4300/4300 [==============================] - 0s 76us/step - loss: 0.0889 - acc: 0.9942 - val_loss: 0.4452 - val_acc: 0.8693

```

```
python feature_extraction.py --training_file inception_traffic_100_bottleneck_features_train.p --validation_file inception_traffic_bottleneck_features_validation.p

Epoch 50/50
4300/4300 [==============================] - 1s 131us/step - loss: 0.0273 - acc: 1.0000 - val_loss: 0.8361 - val_acc: 0.7535

```

```
python feature_extraction.py --training_file resnet_traffic_100_bottleneck_features_train.p --validation_file resnet_traffic_bottleneck_features_validation.p

Epoch 50/50
4300/4300 [==============================] - 1s 127us/step - loss: 0.0320 - acc: 1.0000 - val_loss: 0.6104 - val_acc: 0.8089

```
