import torch
print(torch.__version__) #1.8.2+cu111
import torchvision
print(torchvision.__version__) #0.9.2+cu111
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
print(train_on_gpu)

import detectron2 