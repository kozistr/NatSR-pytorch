# NatSR-pytorch

Unofficial implementation of natural and Realistic Single Image Super-Resolution with Explicit Natural Manifold Discrimination (CVPR, 2019) in pytorch (w/ audit-friendly code)

* official **tensorflow** implementation : [https://github.com/JWSoh/NatSR](https://github.com/JWSoh/NatSR)
* paper : [CVPR2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Soh_Natural_and_Realistic_Single_Image_Super-Resolution_With_Explicit_Natural_Manifold_CVPR_2019_paper.pdf)

**Work In Progress (WIP)**

## Environments

* Python 3.x (recommended 3.7)
* Pytorch 1.x

## Abstract

Recently, many convolutional neural networks for single image super-resolution (SISR) have been proposed, which focus on reconstructing the high-resolution images in terms of objective distortion measures. 
**However**, the networks trained with objective loss functions generally fail to reconstruct the realistic fine textures and details that are essential for better perceptual quality. 
Recovering the realistic details remains a challenging problem, and only a few works have been proposed which aim at increasing the perceptual quality by generating enhanced textures. 
**However**, the generated fake details often make undesirable artifacts and the overall image looks somewhat unnatural. 
**Therefore**, in this paper, we present a new approach to reconstructing realistic super-resolved images with high perceptual quality, while maintaining the naturalness of the result. 
*In particular*, we focus on the domain prior properties of SISR problem. 
Specifically, we define the naturalness prior in the low-level domain and constrain the output image in the natural manifold, which eventually generates more natural and realistic images. 
Our results show better naturalness compared to the recent super-resolution algorithms including perception-oriented ones.

## DataSet

* DIV2K : [download](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

## Usage

0. **Clone** the repository

```shell script
$ git clone https://github.com/kozistr/NatSR-pytorch
$ cd ./NatSR-pytorch
```

1. **Configure** your own environment.

1.1. Using `pipenv` by given *Pipfile*

```shell script
$ pip3 install -U pipenv
$ pipenv install --dev
```

1.2. Using `requirements.txt`

```shell script
$ pip3 install -r requirements.txt
```

2. **Change** the parameter what you want [`config.yaml`](./config.yaml)

2.1. Mode

* At train : `mode: train`
* At test : `mode: test`
* At inference : `mode: inference`

2.2. Model Type

* training *NMD* : `model_type: 'nmd'`
* training *FRSR* : `model_type: 'frsr'`

3. Run!

```shell script
$ python3 -m natsr
```

## Result

## Citation

```
@InProceedings{Soh_2019_CVPR,
  author = {Soh, Jae Woong and Park, Gu Yong and Jo, Junho and Cho, Nam Ik},
  title = {Natural and Realistic Single Image Super-Resolution With Explicit Natural Manifold Discrimination},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

## Author

Hyeongchan Kim / [@kozistr](http://kozistr.tech)
