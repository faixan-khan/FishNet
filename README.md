# FishNet: A Large-scale Dataset and Benchmark for Fish Recognition, Detection, and Functional Traits Prediction(ICCV 2023)

[Faizan Farooq Khan](https://faixan-khan.github.io/), [Xiang Li](https://xiangli.ac.cn/), [Andrew Temple](https://reefecology.kaust.edu.sa/people/details/andrew-temple),  and [Mohamed Elhoseiny](https://www.mohamed-elhoseiny.com/). 


[![demo](fig/teaser-1.png)](https://fishnet-2023.github.io/)


## Introduction

We present FishNet, a comprehensive benchmark for large-scale aquatic species recognition, detection, and functional trait identification. Our benchmark dataset is based on an aquatic biological taxonomy, consisting of 8 taxonomic classes, 83 orders, 463 families, 3,826 genera, 17,357 species, and 94,532 images. The dataset also includes bounding box annotations for fish detection. Additionally, the dataset encompasses 22 traits, grouped into three categories: habitat, ecological rule, and nutritional value. These traits facilitate the identification of the ecological roles of aquatic species and their interactions with other species.


## Dataset Download

To download the required files, follow these steps:

1. **Image Files:**
   All the images can be found [here.](https://drive.google.com/file/d/1mqLoap9QIVGYaPJ7T_KSBfLxJOg2yFY3/view?usp=sharing)

   Change ```args.data_root``` in ```main.py``` file to YOUR_PATH.

2. **Classification and Functional Traits Annotation Files:**
   The files can be found in the ```anns/``` folder.

3. **Detection Annotation Files:**
   Download [bbox.zip] file for detection annotations. 
   We use mmdetction library for fish detection. Please refer to the [tutorial](https://github.com/xy-guo/mmdetection_kitti/blob/dev/demo/MMDet_Tutorial.ipynb) for getting started.


## Fish Classification and Functional Traits Prediction

```python main_v2.py --model mode_id --label_column Order```

**mode_id** denotes the id of vision model to use, should be choosen from below:
```
model mappings
0 : vit_base
1 : vit_small
2 : vit_large
3 : r34
4 : r50
5 : r101
6 : r152
7 : beit
8 : convnext
```

**label_column** denotes the attrubute used for classification, choosen from: 
```
Family: Faimily level classification
Order: Order level classification
MultiCls: Functional traits classification
```

For other parameters, please check ```main.py``` file.


