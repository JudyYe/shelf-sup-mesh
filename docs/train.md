## Train your own models

After setting up the [dataset](dataset.md), we are ready to train our model. 
Among the two steps, only category-level volumetric reconstruction requires training. 
The reconstructed voxels are used to properly initialize the meshes later. 

```
python train_test.py     --dataset allChair --cfg_file config/pmBigChair.json 
```

where `--dataset` specifies the training dataset. Please see [`code`](data/dataset.py#L12) for available choice.  
`--cfg_file` sets configuration specific to each dataset, e.g. elevation range. 
Our method depends on  prior distribution of camera viewpoints, which is manually specified.        

| dataset | `--dataset` |` --cfg_file` | elevation |
| --- | --- | --- | --- |
| OpenImages | oi$subsetname | wild.json| (-30, 30) |
|Chairs in the wild | allCahir | pmBigChair.json| (-5, 30) |
|Quadrupeds | imAll / im$subsetname | quad.json | (-60, 60) |
|CUB-200-2011 | cub | cub.json | (-60, 60) |
|Shapenet | snChair/Cars/Aero | chair.json | (-60, 60) | 



Some side notes:
- **Architecture:** Network architecture is not our focus. So we just use whatever came handy. 
  The reconstruction network follows an encoder-decoder architecture where the encoder is based on a resnet-18 and the decoder  
  follows [HoloGAN](https://github.com/thunguyenphuoc/HoloGAN) 's  architecture. 
  Note that we do not build on the standard StyleGAN implementation in [Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
  as it was not released when we started development. Later I tried to add 3D structure to the official StyleGAN2 repo and made it work on ShapeNet dataset.
- **GAN Training**: Training GAN is notoriously painful.  I have wasted a bunch of time on praying GAN magically trains after 10k iters -- but it will not. After 1k~2k iters, you should get a rough shape like this.    
  ![](2000_azvoxM.gif)   
  Please consider stopping to debug if the 3D does not emerge after 2k iters.
  I also tried some tricks which does not help significantly: 1) WGAN-GP, 2) spectral norm, 3) patch discriminator, 4) more loss term balance etc.
  Another issue we suffer is that GAN collapses with small datasets. Some recent works like [DiffAug](https://github.com/mit-han-lab/data-efficient-gans), 
  or [StyleGAN2-Ada](https://github.com/NVlabs/stylegan2-ada-pytorch) may come to rescue but we did not explore this.   
- Training quadrupeds. it is very challenging to train on this challenging dataset 
due to large shape variance and low quality of segmentations. 
Therefore, we adapt these two changes in the first training step. 
    1. A novel view is biased to sample from two sides instead of from uniform  to be more consistent with the actual bias in the dataset. `--sample_view side`     
    2. Adding a prior towards "mean" voxel [(link)](https://drive.google.com/file/d/1mAUiJtApEkgwxGitrKhV-8IEWN1Rt6va/view?usp=sharing) `--know_vox 1`. It stablizes training and avoids degenerate solutions. 

## Mesh Refinement 
After predicting coarse shape and camera viewpoint, we convert the voxel to a mesh 
and optimize it with respect to reprojection loss and surface smoothness.  

```angular2html
bash scripts/test_opt.sh $GPU $MODEL $ITER 
```

---
## Baselines
With some config modification, baselines can be implemented as well.
For PrGAN, append the following flags :
```angular2html
--d_loss_rgb 0 --d_loss_mask 1 --cyc_loss 0 --cyc_perc_loss 0
```
The flags turn off appearance cue (`--d_loss_rgb 0 --cyc_loss 0  --cyc_perc_loss 0`), and instead apply adversarial loss on masks (`--d_loss_mask 1`).

For HoloGAN, append the following flags:
```angular2html
--vol_render rgb   --mask_loss_type l1 
```
The flags change the renderer to geometric-agnostic one which learns to composite feature points at different depths on a ray. 
As the projection from feature to RGBA is learned, the masks are not guarantee within 0 and 1, which IoU loss requires. Thus we use l1 loss on masks instead.   

More examples of training commands can be found at [`scripts/exp.sh`](../scripts/exp.sh)
