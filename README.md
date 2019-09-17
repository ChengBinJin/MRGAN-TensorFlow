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
  <img src="https://user-images.githubusercontent.com/37034031/64340013-29c7c680-d020-11e9-8fe8-e618338a923f.jpg" width=700)
</p>  

The figure shows an input CT image, and the corresponding synthesized MR images from the CycleGAN and MR-GAN. It also shows their reconstructed CT images and their relative difference maps.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/64340046-39470f80-d020-11e9-8e2b-9e44cb78f117.jpg" width=600)
</p>  
  
## Implementations
### Related Works
- [Paired Training](https://arxiv.org/pdf/1707.09747.pdf)
- [Unpaired Training](https://arxiv.org/pdf/1708.01155.pdf)

## Documentation
### Directory Hierarchy
``` 
MRGAM
├── src
│   ├── DC2Anet
│   │   ├── build_data.py
│   │   ├── cycle_gan.py
│   │   ├── data_preprocess.py
│   │   ├── dataset.py
│   │   ├── DB05_data_preprocessing.py
│   │   ├── dcgan.py
│   │   ├── eval.py
│   │   ├── experiments.py
│   │   ├── gan_repository.py
│   │   ├── laplotter.py
│   │   ├── main.py
│   │   ├── mnist_dataset.py
│   │   ├── mrigan.py
│   │   ├── mrigan_01.py
│   │   ├── mrigan_02.py
│   │   ├── mrigan01_lsgan.py
│   │   ├── mrigan02.py
│   │   ├── mrigan02_lsgan.py
│   │   ├── mrigan03.py
│   │   ├── mrigan03_lsgan.py
│   │   ├── pix2pix.py
│   │   ├── pix2pix_patch.py
│   │   ├── reader.py
│   │   ├── solver.py
│   │   ├── TensorFlow_utils.py
│   │   ├── test.py
│   │   ├── test_main.py
│   │   ├── test_solver.py
│   │   ├── utils.py
│   │   ├── vanila_gan.py
│   │   └── wgan.py
Data
├── brain01
│   └── raw
└── brain02
│   ├── CT.tfrecords
│   └── MRI.tfrecords
```  

### Training MR-GAN
Under the folder of the src, using `main.py` train a DC2Anet model. Example usage:
```
python main.py 
```
- `gan_model`: select gan model, default: `pix2pix`
- `is_train`: training or inference mode, default: `False`
- `is_continue`: continue training, default: `False`
- `gpu_index`: gpu index if you have multiple gpus, default: `0`
- `dataset`: dataset name, default: `brain01`  
- `which_direction`: AtoB (0) or BtoA (1), default: `0`
- `sample_batch`: sample batch size, default: `4`
- `batch_size`: batch size, default: `4`
- `learning_rate`: initial learning rate for Adam, default: `2e-4` 
- `z_dim`: dimension of z vector, default: `100`
- `beta1`: momentum term of Adam, default: `0.5`
- `iters`: number of iterations, default: `200000` 
- `save_freq`: save frequency for model, default: `10000`
- `print_freq`: print frequency for loss, default: `100` 
- `sample_freq`: sample frequency for saving image, default: `500`
- `load_model`: folder of saved model that you wish to continue training, (e.g. 20181127-2116), default: `None`  

### Test MR-GAN

### Algorithm
The MR-GAN procedure is described in the following Algorithm:
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/64615263-2ca62b00-d415-11e9-9aa5-6c6ed089d756.png" width=700)
</p>  

### Comparison with Baselines
Table 1 shows a quantitative evaluation using MAE and PSNR to compare the different methods in the test set. The proposed method is compared with the independent training using the paired and unpaired data.
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/64614965-89551600-d414-11e9-9105-2adb127e7133.png" width=800)
</p>
