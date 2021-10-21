## OpenImages
- Download images of a particular categories from the official [website](https://storage.googleapis.com/openimages/web/index.html) to `data/openimages/JPEGIMages`.
- Download masks of the images and put them under `data/openimages/Segs/`
- Download necessary meta data like mapping between wnid and class name([`class-descriptions-boxable.csv`](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv)).   
- Filter masks.  Masks in OpenImages may be occluded or truncated, which our method does not handle. We filter them out by a  finetuned classifier automatically.
You are welcome to train your own models but we provide the predicted scores from our model [here](https://drive.google.com/file/d/1myeDBjfCTF8LawyORvitUHJTROrcPW4g/view?usp=sharing). We just appended one additional column `truncated` as the quality score (-1 means N/A) to the original OpenImages [meta data](https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv). 
  
The data directory is expected of the following structure:
```
data/
  openimages/
    class-descriptions-boxable.csv  # wnid to class name 
    JPEGImages/*  # Images from website
    Segs/
      Masks/*  # foregroumd masks
      train\/val-annotations-object-segmentation.csv  # Train/validation Mask data from official website  
      train\/val-obj-seg-occBal.csv # auto trucation score, download from here: https://drive.google.com/file/d/1myeDBjfCTF8LawyORvitUHJTROrcPW4g/view?usp=sharing
```

## Chairs in the wild
Chairs in the wild consists of two subset: 1) Stanford online dataset and PASCAL 3D+ chairs.    
### Setup Stanford Online Dataset
- Download [Stanford Online Dataset](https://cvgl.stanford.edu/projects/lifted_struct/) and place it at `data/ebay`
- Download [cached segmentation](https://drive.google.com/file/d/1IpYhJYOXfS_VJOCkW9c4lwIfVRFVxzp7/view?usp=sharing) with split list and extract it at `data/ebay`. The annotation is a copy from [this cool paper](https://shubhtuls.github.io/mvcSnP/)

### Setup PASCAL3D+ 
- Download [PASCAL3D+ release 1.1](https://cvgl.stanford.edu/projects/lifted_struct/) to `data/p3d`
- Download [cached segmentation](https://drive.google.com/file/d/1lDaO_uZrptKylwej-3b0NE7VA3YZVeXu/view?usp=sharing) to `data/p3d/segs` for quick usage. 
- (Optional) Create your own segmentations: big part of code is from [here](https://github.com/akanazawa/cmr/tree/p3d/preprocess/pascal)
    + Set up [PointRend](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend) 
    and run the off-the-shelf model on ImageNet dataset. An example code can be found at [`scripts/pointrend.py`](../scripts/pointrend.py)
    + combine it with PASCAL segmentation annotation and reformat to ready-to-run mat by [`scripts/my_preprocess.m`](../scripts/my_preprocess.m).   

The data directory  is expected to have the with structure:
```
data/
    ebay/
        images/chair_final/*.JPG
        segs/chair/*.JPG
        train.txt
        val.txt
    p3d/
        Imgaes/*
        segs/*
    ...
``` 

---

## CUB-200-2011
Please follow [this](https://github.com/akanazawa/cmr/blob/master/doc/train.md)
A direct link to the cache we used by following their intruction can be found [here](https://drive.google.com/file/d/17w5Aq6IH64p9nntqZOMdQcTaF0Wl4JeO/view?usp=sharing).
 
## Quadrupeds
Please follow [this](https://github.com/nileshkulkarni/acsm/tree/master/quadruped_data) to setup data.


## ShapeNet
Our rendered images can be found [here](https://drive.google.com/file/d/1vASrBZzeaHgwDUoUb3G9EtsCQ6qEsoXx/view?usp=sharing).
