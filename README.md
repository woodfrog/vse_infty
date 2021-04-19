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

The following tables show partial results of image-to-text retrieval on COCO and Flickr30K datasets. In these experiments, we use BERT-base as the text encoder for our methods. This branch provides the code and pre-trained models for using **BiGRU as the text backbone**, please check out to the **```master``` branch** for the code and pre-trained models for using BERT as the text backbone.

#### Results of 5-fold evaluation on COCO 1K Test Split

| | Backbone|R1|R5|R1|R5|Link|
|---|---|---|---|---|---|---|
|SCAN| BUTD region	|72.7|94.8|58.8|88.4| - |
|VSRN (Ensemble) | BUTD region	|76.2|94.8|62.8|89.7| - |
|VSEInfty | BUTD region |**78.5**|**96.0**|**61.7**|**90.3**|[Here]()|
|VSEInfty | BUTD grid |**78.0**|**95.8**|**62.6**|**90.6**|[Here]()|
|VSEInfty (Ensemble) | BUTD region + grid |**80.0**|**97.0**|**64.8**|**91.6**| - |

Note that the last raw is the ensemble of the models from the 3rd and 4th row.

#### Results on Flickr30K Test Split

| | Backbone|R1|R5|R1|R5|Link|
|---|---|---|---|---|---|---|
|SCAN| BUTD region	|67.4|90.3|48.6|77.7| - |
|VSRN (Ensemble)| BUTD region	|71.3|90.6|54.7|81.8| - |
|VSEInfty | BUTD region |**76.5**|**94.2**|**56.4**|**83.4**|[Here]()|
|VSEInfty | BUTD grid |**77.9**|**93.7**|**57.5**|**83.4**|[Here]()| 
|VSEInfty (Ensemble) | BUTD region + grid|**80.7**|**96.4**|**60.8**|**86.3**| - |


## Preparation

### Environment

We trained and evaluated our models with the following key dependencies:

- Python 3.7.3 

- Pytorch 1.2.0

- nltk 3.5


Run ```pip install -r requirements.txt ``` to install the exactly same dependencies as our experiments. However, we also verified that using the latest Pytorch 1.8.0 can also produce similar results.  

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

The ```id_mapping.json``` files are the mapping from image index (ie, the COCO id for COCO images) to corresponding filenames, we generated these mappings to eliminate the need of the ```pycocotools``` package. 

```weights/original_updowmn_backbone.pth``` is the pre-trained ResNet-101 weights from [Bottom-up Attention Model](https://github.com/peteanderson80/bottom-up-attention), we converted the original Caffe weights into Pytorch.

## Training

We provide example training scripts for:

1. Grid feature with BUTD CNN for the image feature, BERT-base for the text feature. See ```train_grid.sh```

2. BUTD Region feature for the image feature, BERT-base for the text feature. See ```train_region.sh```

To use other CNN initializations for the grid image feature, change the ```--backbone_source``` argument to different values: 

- (1). the default ```detector``` is to use the [BUTD ResNet-101](https://github.com/peteanderson80/bottom-up-attention), we have adapted the original Caffe weights into Pytorch and provided the download link above; 
- (2). ```wsl```  is to use the backbones from [large-scale weakly supervised learning](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/); 
- (3). ```imagenet_res152``` is to use the ResNet-152 pre-trained on ImageNet. 



## Evaluation

Run ```eval.py``` to evaluate specified models on either COCO and Flickr30K. 


