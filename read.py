import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

   def __init__(self, x, y):
      self.x = torch.from_numpy(x).float()
      self.y = torch.from_numpy(y)
   

   def __len__(self):
      return len(self.x)
   
   def __getitem__(self, idx):
      return self.x[idx,:], self.y[idx,:]
