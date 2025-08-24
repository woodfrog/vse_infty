# Learning the Best Pooling Strategy for Visual Semantic Embedding

<img src="docs/assets/img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Official PyTorch implementation of the paper [Learning the Best Pooling Strategy for Visual Semantic Embedding](https://arxiv.org/abs/2011.04305) (**CVPR 2021 Oral**).

Please use the following bib entry to cite this paper if you are using any resources from the repo.

```
@inproceedings{chen2021vseinfty,
     title={Learning the Best Pooling Strategy for Visual Semantic Embedding},
     author={Chen, Jiacheng and Hu, Hexiang and Wu, Hao and Jiang, Yuning and Wang, Changhu},
     booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
     year={2021}
} 
```


We referred to the implementations of [VSE++](https://github.com/fartashf/vsepp) and [SCAN](https://github.com/kuanghuei/SCAN) to build up our codebase. 


## Introduction

<img src="docs/assets/img/overview.png" width="100%">

Illustration of the standard Visual Semantic Embedding (VSE) framework with the proposed pooling-based aggregator, i.e., Generalized
Pooling Operator (GPO). It is simple and effective, which automatically adapts to the appropriate pooling strategy given different data
modality and feature extractor, and improves VSE models at negligible extra computation cost.


### Image-text Matching Results

The following tables show partial results of image-to-text retrieval on COCO and Flickr30K datasets. In these experiments, we use BERT-base as the text encoder for our methods. This branch provides our code and pre-trained models for **using BERT as the text backbone**, please check out to [**the ```bigru``` branch**](https://github.com/woodfrog/vse_infty/tree/bigru) for the code and pre-trained models for using BiGRU as the text backbone.

Note that the VSE++ entries in the following tables are the VSE++ model with the specified feature backbones, thus the results are different from the original VSE++ paper.  

#### Results of 5-fold evaluation on COCO 1K Test Split

| |Visual Backbone|Text Backbone|R1|R5|R1|R5|Link|
|---|:---:|:---:|---|---|---|---|---|
|VSE++| BUTD region |BERT-base|67.9|91.9|54.0|85.6| - |
|VSEInfty | BUTD region |BERT-base|**79.7**|**96.4**|**64.8**|**91.4**|[Here](https://huggingface.co/cccjc/vse-infty/tree/main/coco_butd_region_bert)|
|VSEInfty | BUTD grid |BERT-base|**80.4**|**96.8**|**66.4**|**92.1**|[Here](https://huggingface.co/cccjc/vse-infty/tree/main/coco_butd_grid_bert)|
|VSEInfty | WSL grid |BERT-base|**84.5**|**98.1**|**72.0**|**93.9**|[Here](https://huggingface.co/cccjc/vse-infty/blob/main/coco_wsl_grid_bert/)|

#### Results on Flickr30K Test Split

| |Visual Backbone|Text Backbone|R1|R5|R1|R5|Link|
|---|:---:|:---:|---|---|---|---|---|
|VSE++| BUTD region |BERT-base|63.4|87.2|45.6|76.4|- |
|VSEInfty | BUTD region |BERT-base|**81.7**|**95.4**|**61.4**|**85.9**|[Here](https://huggingface.co/cccjc/vse-infty/tree/main/f30k_butd_region_bert/)|
|VSEInfty | BUTD grid |BERT-base|**81.5**|**97.1**|**63.7**|**88.3**|[Here](https://huggingface.co/cccjc/vse-infty/blob/main/f30k_butd_grid_bert/)| 
|VSEInfty | WSL grid |BERT-base|**88.4**|**98.3**|**74.2**|**93.7**|[Here](https://huggingface.co/cccjc/vse-infty/blob/main/f30k_wsl_grid_bert/)|

#### Result (in R@1) on Crisscrossed Caption benchmark (trained on COCO)

| |Visual Backbone|Text Backbone|I2T|T2I|T2T|I2I|
|---|:---:|:---:|---|---|---|---|
|[VSRN](https://arxiv.org/pdf/1909.02701.pdf) | BUTD region |BiGRU| 52.4 | 40.1 | 41.0 | 44.2 | 
|[DE](https://arxiv.org/abs/2004.15020) |  EfficientNet-B4 grid |BERT-base|55.9|41.7|42.6|38.5| 
|VSEInfty | BUTD grid |BERT-base|60.6| 46.2 | 45.9 | 44.4 | 
|VSEInfty | WSL grid |BERT-base| 67.9 | 53.6 | 46.7 | 51.3 |


## Preparation

### Environment

We trained and evaluated our models with the following key dependencies:

- Python 3.7.3 

- Pytorch 1.2.0

- Transformers 2.1.0


Run ```pip install -r requirements.txt ``` to install the exactly same dependencies as our experiments. However, we also verified that using the latest Pytorch 1.8.0 and Transformers 4.4.2 can also produce similar results.  

### Data

We organize all data used in the experiments in the following manner:

```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── images   # raw coco images
│   │      ├── train2014
│   │      └── val2014
│   │
│   ├── cxc_annots # annotations for evaluating COCO-trained models on the CxC benchmark
│   │
│   └── id_mapping.json  # mapping from coco-id to image's file name
│   
│
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── flickr30k-images   # raw coco images
│   │      ├── xxx.jpg
│   │      └── ...
│   └── id_mapping.json  # mapping from f30k index to image's file name
│   
├── weights
│      └── original_updown_backbone.pth # the BUTD CNN weights
│
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)
```

The download links for original COCO/F30K images, precomputed BUTD features, and corresponding vocabularies are from the offical repo of [SCAN](https://github.com/kuanghuei/SCAN#download-data). The ```precomp``` folders contain pre-computed BUTD region features, ```data/coco/images``` contains raw MS-COCO images, and ```data/f30k/flickr30k-images``` contains raw Flickr30K images. 

(Update: It seems that the download link for the pre-computed features in SCAN's repo is down, [this Dropbox link](https://www.dropbox.com/sh/qp3fw9hqegpm914/AAC3D3kqkh5i4cgZOfVmlWCDa?dl=0) provides a copy of these files. Please download and follow the above file structures to organize the data.)

The ```id_mapping.json``` files are the mapping from image index (ie, the COCO id for COCO images) to corresponding filenames, we generated these mappings to eliminate the need of the ```pycocotools``` package. 

```weights/original_updowmn_backbone.pth``` is the pre-trained ResNet-101 weights from [Bottom-up Attention Model](https://github.com/peteanderson80/bottom-up-attention), we converted the original Caffe weights into Pytorch. Please download it from [this link](https://drive.google.com/file/d/1gNdV1Qx_7yYzkhHrzqbP-bbNkdrKw_w1/view?usp=sharing).


The ```data/coco/cxc_annots``` directory contains the necessary data files for running the [Criscrossed Caption (CxC) evaluation](https://github.com/google-research-datasets/Crisscrossed-Captions). Since there is no official evaluation protocol in the CxC repo, we processed their raw data files and generated these data files to implement our own evaluation.  We have verified our implementation by aligning the evaluation results of [the official VSRN model](https://github.com/KunpengLi1994/VSRN) with the ones reported by the [CxC paper](https://arxiv.org/abs/2004.15020) Please download the data files at [this link](https://drive.google.com/drive/folders/1Ikwge0usPrOpN6aoQxsgYQM6-gEuG4SJ?usp=sharing).

Please download all necessary data files and organize them in the above manner, the path to the ```data``` directory will be the argument to the training script as shown below.

## Training

Assuming the data root is ```/tmp/data```, we provide example training scripts for:

1. Grid feature with BUTD CNN for the image feature, BERT-base for the text feature. See ```train_grid.sh```

2. BUTD Region feature for the image feature, BERT-base for the text feature. See ```train_region.sh```


To use other CNN initializations for the grid image feature, change the ```--backbone_source``` argument to different values: 

- (1). the default ```detector``` is to use the [BUTD ResNet-101](https://github.com/peteanderson80/bottom-up-attention), we have adapted the original Caffe weights into Pytorch and provided the download link above; 
- (2). ```wsl```  is to use the backbones from [large-scale weakly supervised learning](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/); 
- (3). ```imagenet_res152``` is to use the ResNet-152 pre-trained on ImageNet. 



## Evaluation

Run ```eval.py``` to evaluate specified models on either COCO and Flickr30K. For evaluting pre-trained models on COCO, use the following command (assuming there are 4 GPUs, and the local data path is /tmp/data):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset coco --data_path /tmp/data/coco
```

For evaluting pre-trained models on Flickr-30K, use the command: 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset f30k --data_path /tmp/data/f30k
```

For evaluating pre-trained COCO models on the CxC dataset, use the command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset coco --data_path /tmp/data/coco --evaluate_cxc
```


For evaluating two-model ensemble, first run single-model evaluation commands above with the argument ```--save_results```, and then use ```eval_ensemble.py``` to get the results (need to manually specify the paths to the saved results). 



