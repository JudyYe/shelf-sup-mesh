python train_test.py  \
    --exp repold  --dataset allChair  --cfg_file config/pmBigChair.json  \
    --seed 1234 \
    --gpu 0

python train_test.py  \
    --exp repold  --dataset allChair  --cfg_file config/pmBigChair.json  \
    --gpu 0


python train_test.py  \
    --exp repold  --dataset cub  --cfg_file config/cub.json  \
    --gpu




# Chairs in the wild
#ours
python train_test.py  \
    --exp real  --cfg_file config/pmBigChair.json  \
    --dataset allChair \
    --gpu 0

#HoloGAN
python train_test.py  \
    --exp real  --cfg_file config/pmBigChair.json  \
    --dataset allChair \
    --vol_render rgb   --mask_loss_type l1 \
    --gpu 0

#prGAN
python train_test.py  \
    --exp real  --cfg_file config/pmBigChair.json  \
    --dataset allChair \
    --d_loss_rgb 0 --d_loss_mask 1 --cyc_loss 0 --cyc_perc_loss 0 --lr 1e-5 \
    --gpu 0


# Quadruped
#ours
python train_test.py  \
    --exp real  --cfg_file config/quad.json  \
    --dataset imAll     --know_mean 1 \
    --gpu 0


#HoloGAN
python train_test.py  \
    --exp real  --cfg_file config/quad.json  \
    --dataset imAll     --know_mean 1 \
    --vol_render rgb   --mask_loss_type l1 \
    --gpu 0

#prGAN
python train_test.py  \
    --exp real  --cfg_file config/quad.json  \
    --dataset imAll     --know_mean 1 \
    --d_loss_rgb 0 --d_loss_mask 1 --cyc_loss 0 --cyc_perc_loss 0 --lr 1e-5 \
    --gpu 0