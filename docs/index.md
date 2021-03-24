---
layout: default
title: Learning the Best Pooling Strategy for Visual Semantic Embedding
---

# Abstract

Visual Semantic Embedding (VSE) is a dominant approach for vision-language retrieval, which aims at learning a deep embedding space such that visual data are embedded close to their semantic text labels or descriptions. Recent VSE models use complex methods to better contextualize and aggregate multi-modal features into holistic embeddings. However, we discover that surprisingly simple (but carefully selected) global pooling functions (e.g., max pooling) outperform those complex models, across different feature extractors. Despite its simplicity and effectiveness, seeking the best pooling function for different data modality and feature extractor is costly and tedious, especially when the size of features varies (e.g., text, video). Therefore, we propose a Generalized Pooling Operator (GPO), which learns to automatically adapt itself to the best pooling strategy for different features, requiring no manual tuning while staying effective and efficient. We extend the VSE model using this proposed GPO and denote it as *VSE∞*.

Without bells and whistles, *VSE∞* outperforms previous VSE methods significantly on image-text retrieval benchmarks across popular feature extractors. With a simple adaptation, variants of *VSE∞* further demonstrate its strength by achieving the new state of the art on two video-text retrieval datasets. Comprehensive experiments and visualizations confirm that GPO always discovers the best pooling strategy and can be a plug-and-play feature aggregation module for standard VSE models. 

# Paper

<div>
	<a href="assets/xxx.pdf">
	<img class="thumbnail" src="assets/img/xxx.png"> 
	</a>
</div>>

<div class="text-center">
	<a href="assets/xxxx.pdf"> Download PDF </a> &nbsp; &nbsp; <a href="https://arxiv.org/abs/2011.04305"> Arxiv </a> &nbsp; &nbsp; <a href="assets/xxxx.pdf"> Supplementary </a>
</div>

<br>
<div class="bibtex-box">
	<strong>@InProceedings{</strong>chen2021vseinfty,
	<br>
	&nbsp;&nbsp;&nbsp;&nbsp; title={Learning the Best Pooling Strategy for Visual Semantic Embedding}, 
	<br> 
	&nbsp;&nbsp;&nbsp;&nbsp; author={Jiacheng Chen, Hexiang Hu, Hao Wu, Yuning Jiang, Changhu Wang},
	<br> 
	&nbsp;&nbsp;&nbsp;&nbsp; booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	<br> 
	&nbsp;&nbsp;&nbsp;&nbsp; year={2021}<br><strong>}</strong>
</div>


# Code / Data

Check our code and model checkpoints on our [Github repo](https://github.com/woodfrog/vse_infty). 


# Video

<!-- <div>
<iframe width="820" height="492" src="https://www.youtube.com/embed/PyYz7XAs7UE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
</iframe>
</div> -->


<!-- # Acknowledgement -->
