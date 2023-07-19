# Evaluation code for REID-CBD
### Requirements
numpy>=1.21.5  
torchvision>=0.14.1+cu117  
torch>=1.13.1+cu117  
Pillow>=9.0.1  

### Example
To evaluate checkpoint https://github.com/wuancong/REID-CBD/evaluation/checkpoint/REID-CBD_checkpoint.pth.tar on REID-CBD, run
~~~
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--checkpoint_path checkpoint/REID-CBD_checkpoint.pth.tar \
--dataset_name REID-CBD \
--dataset_dir /path/to/REID-CBD \
--query_list /path/to/REID-CBD/query.txt \
--gallery_list /path/to/REID-CBD/gallery.txt \
~~~

### Acknowledgement
The evaluation code is based on Torchreid https://github.com/KaiyangZhou/deep-person-reid.
