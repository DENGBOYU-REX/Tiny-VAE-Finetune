CUDA_VISIBLE_DEVICES=7 \
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="localhost" --master_port=12334 \
 distribute_train.py \
 --output_dir=outputs/VAE \
 --dataset='/data2/workspace/bydeng/DATASETS/FOR_ERASER/mix_data/train/label' \
 --batch_size=4 \
 --num_epochs=100 \
 --image_size=512 \
 --lr=5e-6 \
 --save_steps=15000 \
 --load_resume='/data2/workspace/bydeng/Projects/VAE/weights/keep_finetune/12_4000' \
 