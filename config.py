import torch

Device = "cuda" if torch.cuda.is_available() else "cpu"

#### 需要自己调
Epochs = 1000
Image_size = 64
Dataset_path = "dataset_human"
Image_format = "jpg"
Batch_size = 16

Save_checkpoint_interval = 10

Load_weights = True

# 多线程处理数据集，windows下最好为0
Num_workers = 0
####

Style_mixing_prob = 0.9


D_latent = 512
Mapping_network_layers = 8
Mapping_network_learning_rate = 1e-5
Learning_rate = 1e-3
Adam_betas = (0.0, 0.99)
Gradient_accumulate_steps = 5
Lazy_gradient_penalty_interval = 4
Gradient_penalty_coefficient = 10.
Lazy_path_penalty_after = 40


