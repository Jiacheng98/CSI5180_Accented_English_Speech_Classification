import torch
from torch import nn
torch.manual_seed(1)

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, num_frame, num_mfcc_feature, num_class):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(num_mfcc_feature * num_frame, 32),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      nn.Linear(32, 16),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      nn.Linear(16, num_class)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)