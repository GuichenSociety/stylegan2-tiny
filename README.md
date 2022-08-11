
# stylegan2-tiny

It is more concise than stylegan2. If you want to learn stlegan network, you can come in and learn it.比stylegan2更简洁，如果是想学习stlegan网络可以进来学习一下。


## Installation

Install project

```bash
conda create -n stylegan2-tiny python 3.x
conda activate stylegan2-tiny
cd stylegan2-tiny
pip install -r requirements.txt
```
    
## Train
Enter the config file and modify it.进入config文件下进行修改。
```bash
Epochs = 1000                                    Training times.训练次数。
Image_size = 64                                  Enter or generate picture size.输入或生成图片大小。
Dataset_path = "dataset_human"                   Data set address, subfolder containing pictures.数据集地址，子文件夹包含图片。
Image_format = "jpg"                             
Batch_size = 16                                  Training batch of data set.数据集的训练批次。
Load_weights = True
```
## 🔗 Links
[Referenced articles.参考的文章。](https://nn.labml.ai/gan/stylegan/index.html)
[stylegan3](https://github.com/NVlabs/stylegan3)
