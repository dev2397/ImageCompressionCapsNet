# Image Compression using Capsule Networks

TensorFlow Implementation for JPEG Image compression using Capsule Networks. The original idea of Capsule Networks was developed by Sara Sabour et. al. in [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829). We are going to be using Justin Tan's [implementation](https://github.com/Justin-Tan/generative-compression) of the paper [Generative Adversarial Networks for Extreme Learned Image Compression](https://arxiv.org/abs/1804.02958), developed by Agustsson et. al. , and modify it to work with capsule networks. The CapsLayer object has been used from [naturomics capsule network repository](https://github.com/naturomics/CapsNet-Tensorflow). 

We have attempted to replace the discriminator in the GAN with a capsule network, given by `capsule_discriminator.py`. An ongoing project, results of which would be published very soon. We are using the cityscapes `leftimg8bit` dataset for testing purposes. 

