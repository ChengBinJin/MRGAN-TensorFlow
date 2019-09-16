# MRGAN-TensorFlow
This repository is a TensorFlow implementation of the paper "[Deep CT to MR Synthesis Using Paired and Unpaired Data](https://www.mdpi.com/1424-8220/19/10/2361)," Sensors, 2019, 19(10), 2361.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/64339243-8c1fc780-d01e-11e9-9fba-2f39279fb30f.png")
</p>

## Requirements
- tensorflow 1.14.0
- numpy 1.16.2
- opencv 4.1.0.25
- scipy 1.2.1
- matplotlib 3.0.3

## CT-Based Synthetic MRI Generation Results
The following figure shows a qualitative comparison between the paired training, unpaired training, and the approach presented herein.  

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/64340013-29c7c680-d020-11e9-8fe8-e618338a923f.jpg" width=600)
</p>  

The figure shows an input CT image, and the corresponding synthesized MR images from the CycleGAN and MR-GAN. It also shows their reconstructed CT images and their relative difference maps.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/64340046-39470f80-d020-11e9-8e2b-9e44cb78f117.jpg" width=600)
</p>  

The MR-GAN procedure is described in the following Algorithm:
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/64615263-2ca62b00-d415-11e9-9aa5-6c6ed089d756.png" width=700)
</p>  

Table 1 shows a quantitative evaluation using MAE and PSNR to compare the different methods in the test set. The proposed method is compared with the independent training using the paired and unpaired data.
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/64614965-89551600-d414-11e9-9105-2adb127e7133.png" width=800)
</p>
