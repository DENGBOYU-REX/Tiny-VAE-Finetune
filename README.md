# Hi, this is a sample way to finetune your VAE model for mask inpainting using torchrun.

You need to install following packages:
```
wget https://download.pytorch.org/whl/cu118/torch-2.4.0%2Bcu118-cp311-cp311-linux_x86_64.whl#sha256=6acd608416b12211e21dfe5b92ffb1c82126ee8d037dd119f45d8b28ed80a0d2
pip install torch-2.4.0+cu118-cp311-cp311-linux_x86_64.whl
pip install diffusers==0.31.0
pip install torchvision==0.20.1 torchmetrics=1.6.1
```
Change the params in .sh file for your own model and data.
