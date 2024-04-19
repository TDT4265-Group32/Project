import os
import torch
from tqdm import tqdm

class BaseModel:
    
    def __init__(self):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
