## Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set —— Official PyTorch implementation ##

<p align="center"> 
<img src="/images/example.gif">
</p>

This is an official pytorch implementation of the following paper:

Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, [Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set](https://arxiv.org/abs/1903.08527), IEEE Computer Vision and Pattern Recognition Workshop (CVPRW) on Analysis and Modeling of Faces and Gestures (AMFG), 2019. (**_Best Paper Award!_**)

The method enforces a hybrid-level weakly-supervised training for CNN-based 3D face reconstruction. It is fast, accurate, and robust to pose and occlussions. It achieves state-of-the-art performance on multiple datasets such as FaceWarehouse, MICC Florence and NoW Challenge.

For the original tensorflow implementation, check this [repo](https://github.com/microsoft/Deep3DFaceReconstruction).

## Performance

### ● Reconstruction accuracy

The pytorch implementation achieves lower shape reconstruction error (9% improvement) compare to the [original tensorflow implementation](https://github.com/microsoft/Deep3DFaceReconstruction). Quantitative evaluation (average shape errors in mm) on several benchmarks is as follows:

|Method|FaceWareHouse|MICC Florence     | NoW Challenge |
|:----:|:-----------:|:-----------:|:-----------:|
|Deep3D Tensorflow |  1.81±0.50  |  1.67±0.50  | 1.54±1.29 |
|**Deep3D PyTorch** |**1.64±0.50**|**1.53±0.45**| **1.41±1.21** |

The comparison result with state-of-the-art public 3D face reconstruction methods on the NoW face benchmark is as follows:
|Rank|Method|Median(mm)    | Mean(mm) | Std(mm) |
|:----:|:-----------:|:-----------:|:-----------:|:-----------:|
| 1. | [DECA\[Feng et al., SIGGRAPH 2021\]](https://github.com/YadiraF/DECA)|1.09|1.38|1.18|
| **2.** | **Deep3D PyTorch**|**1.11**|**1.41**|**1.21**|
| 3. | 	[RingNet [Sanyal et al., CVPR 2019]](https://github.com/soubhiksanyal/RingNet) | 1.21 | 1.53 | 1.31 |
| 4. | [Deep3D [Deng et al., CVPRW 2019]](https://github.com/microsoft/Deep3DFaceReconstruction) | 1.23 | 1.54 | 1.29 |
| 5. | [3DDFA-V2 [Guo et al., ECCV 2020]](https://github.com/cleardusk/3DDFA_V2) | 1.23 | 1.57 | 1.39 |
| 6. | [MGCNet [Shang et al., ECCV 2020]](https://github.com/jiaxiangshang/MGCNet) | 1.31 | 1.87 | 2.63 |
| 7. | [PRNet [Feng et al., ECCV 2018]](https://github.com/YadiraF/PRNet) | 1.50 | 1.98 | 1.88 |
| 8. | [3DMM-CNN [Tran et al., CVPR 2017]](https://github.com/anhttran/3dmm_cnn) | 1.84 | 2.33 | 2.05 |

For more details about the evaluation, check [Now Challenge](https://ringnet.is.tue.mpg.de/challenge) website.

### ● Visual quality
The pytorch implementation achieves better visual consistency with the input images compare to the original tensorflow version.

<p align="center"> 
<img src="/images/compare.png">
</p>

### ● Speed
The training speed is on par with the original tensorflow implementation. For more information, see [here](https://github.com/sicxu/Deep3DFaceRecon_pytorch#train-the-face-reconstruction-network).

## Major changes

### ● Differentiable renderer

We use [Nvdiffrast](https://nvlabs.github.io/nvdiffrast/) which is a pytorch library that provides high-performance primitive operations for rasterization-based differentiable rendering. The original tensorflow implementation used [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer) instead.

### ● Face recognition model

We use [Arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch), a state-of-the-art face recognition model, for perceptual loss computation. By contrast, the original tensorflow implementation used [Facenet](https://github.com/davidsandberg/facenet).

### ● Training configuration

Data augmentation is used in the training process which contains random image shifting, scaling, rotation, and flipping. We also enlarge the training batchsize from 5 to 32 to stablize the training process. 

### ● Training data

We use an extra high quality face image dataset [FFHQ](https://github.com/NVlabs/ffhq-dataset) to increase the diversity of training data.

## Requirements
**This implementation is only tested under Ubuntu environment with Nvidia GPUs and CUDA installed.**

## Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git --recursive
cd Deep3DFaceRecon_pytorch
conda env create -f environment.yml
source activate deep3d_pytorch
```

2. Install Nvdiffrast library:
```
cd nvdiffrast    # ./Deep3DFaceRecon_pytorch/nvdiffrast
pip install .
```

3. Install Arcface Pytorch:
```
cd ..    # ./Deep3DFaceRecon_pytorch
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch/ ./models/
```
## Inference with a pre-trained model

### Prepare prerequisite models
1. Our method uses [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) to represent 3d faces. Get access to BFM09 using this [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access, download "01_MorphableModel.mat". In addition, we use an Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace). Download the Expression Basis (Exp_Pca.bin) using this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing). Organize all files into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── BFM
    │
    └─── 01_MorphableModel.mat
    │
    └─── Exp_Pca.bin
    |
    └─── ...
```
2. We provide a model trained on a combination of [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 
[LFW](http://vis-www.cs.umass.edu/lfw/), [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm),
[IJB-A](https://www.nist.gov/programs-projects/face-challenges), [LS3D-W](https://www.adrianbulat.com/face-alignment), and [FFHQ](https://github.com/NVlabs/ffhq-dataset) datasets. Download the pre-trained model using this [link (google drive)](https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP?usp=sharing) and organize the directory into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── checkpoints
    │
    └─── <model_name>
        │
        └─── epoch_20.pth

```

### Test with custom images
To reconstruct 3d faces from test images, organize the test image folder as follows:
```
Deep3DFaceRecon_pytorch
│
└─── <folder_to_test_images>
    │
    └─── *.jpg/*.png
    |
    └─── detections
        |
	└─── *.txt
```
The \*.jpg/\*.png files are test images. The \*.txt files are detected 5 facial landmarks with a shape of 5x2, and have the same name as the corresponding images. Check [./datasets/examples](datasets/examples) for a reference.

Then, run the test script:
```
# get reconstruction results of your custom images
python test.py --name=<model_name> --epoch=20 --img_folder=<folder_to_test_images>

# get reconstruction results of example images
python test.py --name=<model_name> --epoch=20 --img_folder=./datasets/examples
```

Results will be saved into ./checkpoints/<model_name>/results/<folder_to_test_images>, which contain the following files:
| \*.png | A combination of cropped input image, reconstructed image, and visualization of projected landmarks.
|:----|:-----------|
| \*.obj | Reconstructed 3d face mesh with predicted color (texture+illumination) in the world coordinate space. Best viewed in Meshlab. |
| \*.mat | Predicted 257-dimensional coefficients and 68 projected 2d facial landmarks. Best viewd in Matlab.

## Training a model from scratch
### Prepare prerequisite models
1. We rely on [Arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) to extract identity features for loss computation. Download the pre-trained model from Arcface using this [link](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#ms1mv3). By default, we use the resnet50 backbone ([ms1mv3_arcface_r50_fp16](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC)), organize the download files into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── checkpoints
    │
    └─── recog_model
        │
        └─── ms1mv3_arcface_r50_fp16
	    |
	    └─── backbone.pth
```
2. We initialize R-Net using the weights trained on [ImageNet](https://image-net.org/). Download the weights provided by PyTorch using this [link](https://download.pytorch.org/models/resnet50-0676ba61.pth), and organize the file as the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── checkpoints
    │
    └─── init_model
        │
        └─── resnet50-0676ba61.pth
```
3. We provide a landmark detector (tensorflow model) to extract 68 facial landmarks for loss computation. The detector is trained on [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm), [LFW](http://vis-www.cs.umass.edu/lfw/), and [LS3D-W](https://www.adrianbulat.com/face-alignment) datasets. Download the trained model using this [link (google drive)](https://drive.google.com/file/d/1Jl1yy2v7lIJLTRVIpgg2wvxYITI8Dkmw/view?usp=sharing) and organize the file as follows:
```
Deep3DFaceRecon_pytorch
│
└─── checkpoints
    │
    └─── lm_model
        │
        └─── 68lm_detector.pb
```
### Data preparation
1. To train a model with custom images，5 facial landmarks of each image are needed in advance for an image pre-alignment process. We recommend using [dlib](http://dlib.net/) or [MTCNN](https://github.com/ipazc/mtcnn) to detect these landmarks. Then, organize all files into the following structure:
```
Deep3DFaceRecon_pytorch
│
└─── datasets
    │
    └─── <folder_to_training_images>
        │
        └─── *.png/*.jpg
	|
	└─── detections
            |
	    └─── *.txt
```
The \*.txt files contain 5 facial landmarks with a shape of 5x2, and should have the same name with their corresponding images.

2. Generate 68 landmarks and skin attention mask for images using the following script:
```
# preprocess training images
python data_preparation.py --img_folder <folder_to_training_images>

# alternatively, you can preprocess multiple image folders simultaneously
python data_preparation.py --img_folder <folder_to_training_images1> <folder_to_training_images2> <folder_to_training_images3>

# preprocess validation images
python data_preparation.py --img_folder <folder_to_validation_images> --mode=val
```
The script will generate files of landmarks and skin masks, and save them into ./datasets/<folder_to_training_images>. In addition, it also generates a file containing the path of all training data into ./datalist which will then be used in the training script.

### Train the face reconstruction network
Run the following script to train a face reconstruction model using the pre-processed data:
```
# train with single GPU
python train.py --name=<custom_experiment_name> --gpu_ids=0

# train with multiple GPUs
python train.py --name=<custom_experiment_name> --gpu_ids=0,1

# train with other custom settings
python train.py --name=<custom_experiment_name> --gpu_ids=0 --batch_size=32 --n_epochs=20
```
Training logs and model parameters will be saved into ./checkpoints/<custom_experiment_name>. 

By default, the script uses a batchsize of 32 and will train the model with 20 epochs. For reference, the pre-trained model in this repo is trained with the default setting on a image collection of 300k images. A single iteration takes 0.8~0.9s on a single Tesla M40 GPU. The total training process takes around two days.

To use a trained model, see [Inference](https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/main/README.md#inference-with-a-pre-trained-model) section.
## Contact
If you have any questions, please contact Yu Deng (t-yudeng@microsoft.com), Jiaolong Yang (jiaoyan@microsoft.com) or Sicheng Xu (sicheng_xu@yeah.net).

## Citation

Please cite the following paper if this model helps your research:

	@inproceedings{deng2019accurate,
	    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
	    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
	    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
	    year={2019}
	}
##
The face images on this page are from the public [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset released by MMLab, CUHK.
