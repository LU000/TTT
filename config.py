import torch 

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = torch.device('cuda')
SEQ_MAX_LEN=5000