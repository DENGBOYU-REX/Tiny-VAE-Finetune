# -*- encoding: utf-8 -*-
'''
@File    :   distribute_train.py
@Time    :   2024/12/26 11:28:40
@Author  :   DENG Boyu
@Version :   1.0
@Contact :   bydeng_29@163.com
@Desc    :   VAE mdoel distribued training
'''
import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import torch
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
# from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import numpy as np
# from loss import ms_ssim
import random
from torch.utils.tensorboard import SummaryWriter
# 设置设备
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop
from PIL import Image, ImageDraw

# num_epochs = 200
# batch_size = 4
# dir_name = 'VAE_reconstruction'

def args_parser():
    parser = argparse.ArgumentParser(description='VAE distributed training args')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--save_steps', type=int, default=15000)
    parser.add_argument('--load_resume', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    opt = parser.parse_args()
    return opt



# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        # self.mask_dir = mask_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 获取文件名列表
        image_files = set(os.listdir(root_dir))
        # mask_files = set(os.listdir(mask_dir))
        label_files = set(os.listdir(label_dir))
        
        # 找到所有三个目录中都存在的文件名（不包括扩展名）
        common_filenames = set(os.path.splitext(f)[0] for f in image_files) & \
                            set(os.path.splitext(f)[0] for f in label_files)
                        #    set(os.path.splitext(f)[0] for f in mask_files) & \
                           
        
        # 构建完整的文件路径
        self.image_files = [f"{fn}.png" for fn in common_filenames]
        # self.mask_files = [f"{fn}.png" for fn in common_filenames]
        self.label_files = [f"{fn}.png" for fn in common_filenames]
        
        # assert len(self.image_files) == len(self.mask_files) == len(self.label_files), "The number of images, masks, and labels must be the same."
        assert len(self.image_files) == len(self.label_files), "The number of images, masks, and labels must be the same."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        
        image = Image.open(img_path).convert("RGB").resize((800, 800))  # 调整图像大小以匹配VAE输入
        # mask = Image.open(mask_path).convert("L").resize((800, 800))     # 掩码应该是灰度图像
        label = Image.open(label_path).convert("RGB").resize((800, 800))  # 调整图像大小以匹配VAE输入

        image_masked = self.create_random_mask(image)

        return self.transform(image_masked), self.transform(label)
    
    def mask(self, image, mask, label):
        # 将掩码转换为二进制掩码（白色为1，黑色为0）
        binary_mask = (np.array(mask) > 127.5).astype(np.float32)
        
        # 将二进制掩码扩展为三通道
        binary_mask_3ch = np.stack([binary_mask] * 3, axis=-1)
        
        # 将图像和标签转换为numpy数组
        image_np = np.array(image).astype(np.float32)
        label_np = np.array(label).astype(np.float32)
        
        # 应用掩码
        image_masked = image_np * binary_mask_3ch
        label_masked = label_np * binary_mask_3ch
        
        # 转换回PIL图像
        image_masked = Image.fromarray((image_masked).astype(np.uint8))
        label_masked = Image.fromarray((label_masked).astype(np.uint8))
        
        return image_masked, label_masked
    
    def create_random_mask(self, image):
      
        w, h = image.width, image.height
        mask = Image.new('L', (w, h), color=255)
        draw = ImageDraw.Draw(mask)

        # 随机生成一些矩形区域作为mask的部分
        num_rectangles = random.randint(5, 10)  # 随机生成5到10个矩形
        for _ in range(num_rectangles):
            x0 = random.randint(0, w - 1)
            y0 = random.randint(0, h - 1)
            x1 = random.randint(x0 + 1, x0 + w // 5)
            y1 = random.randint(y0 + 1, y0 + h // 5)
            draw.rectangle([x0, y0, x1, y1], fill=0)  

        # mask * image
        mask_image = self.mask_plus_image(image, mask)
        return mask_image

    def mask_plus_image(self, image, mask):
        # 将掩码转换为二进制掩码（白色为1，黑色为0）
        binary_mask = (np.array(mask) > 127.5).astype(np.float32)
        
        # 将二进制掩码扩展为三通道
        binary_mask_3ch = np.stack([binary_mask] * 3, axis=-1)
        
        # 将图像和标签转换为numpy数组
        image_np = np.array(image).astype(np.float32)
        
        # 应用掩码
        image_masked = image_np * binary_mask_3ch
        
        # 转换回PIL图像
        image_masked = Image.fromarray((image_masked).astype(np.uint8))
        
        return image_masked


def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()




def average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def average_loss(loss):
    # Loss averaging.
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= dist.get_world_size()
    return loss

def train():
    opt = args_parser()
    print (opt)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Running basic DDP example on rank {rank}, local rank {local_rank}.")
    # Set the CUDA device for this process
    torch.cuda.set_device(local_rank)
    
    setup(rank, world_size)

    device = local_rank
    ssim_loss =StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse_loss_fn = torch.nn.MSELoss()
    # 加载模型
    vae = AutoencoderKL.from_pretrained(opt.load_resume) # ("/data2/workspace/bydeng/Projects/VAE/weights/keep_finetune_12_23/4_8000")
    print ('load well')
    # vae = SyncBatchNorm.convert_sync_batchnorm(vae)
    vae = vae.to(device)
    vae = DDP(vae, device_ids=[rank], output_device=rank)
    # 微调设置
    optimizer = torch.optim.AdamW(vae.parameters(), lr = opt.lr)#lr=5e-6)

    # 数据加载器
    transform = Compose([
        ToTensor(),
        Resize((opt.image_size, opt.image_size)),
        # RandomCrop((512, 512))
        # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
    ])
    
    # 创建数据集和采样器
    dataset = ImageDataset(opt.dataset,  
                            opt.dataset, transform) # ImageDataset(...)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    num_epochs = opt.num_epochs
    
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader),
    )

    sum_step = 0
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 重要：确保各个进程加载不同数据
        
        if rank == 0:
            progress_bar = tqdm(total=len(dataloader))
        
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播和损失计算
            # with autocast(device_type='cuda', dtype=torch.float16):
            vae_out = vae(images)[0]
            loss_mse = mse_loss_fn(vae_out.float(), labels.float())
            loss_ssim = 1 - ssim_loss(vae_out.float(), labels.float())
            loss_ms_ssim = 1 - ms_ssim(vae_out.float(), labels.float())
            loss = loss_mse * 5 + loss_ssim * 5 + loss_ms_ssim * 5
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # Average the gradients before the optimizer step
            average_gradients(vae)
            
            optimizer.step()
            
            # Average the loss
            loss = average_loss(loss)
            
            lr_scheduler.step()

            sum_step = sum_step + 1
            if rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", mse_loss=f"{loss_mse.item():.4f}", ssim_loss=f"{loss_ssim.item():.4f}", ms_ssim_loss=f"{loss_ms_ssim.item():.4f}")
        
            if sum_step % opt.save_steps == 0:
                # 保存模型
                if rank == 0:
                    save_dir = f'{opt.output_dir}/weights/epoch_{epoch}_step_{sum_step}'
                    os.makedirs(save_dir, exist_ok=True)
                    # 保存未包装的模型
                    vae.module.save_pretrained(save_dir)
        
        # 等待所有进程
        dist.barrier()
    
    cleanup()

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    train()
    # os.environ['CUDA_VISIBLE_DEVICES']='0,6'
    # world_size = torch.cuda.device_count()
    # world_size = 1  # 明确指定world_size为2
    # print(world_size)
    # print(f"Using {world_size} GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    # mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)