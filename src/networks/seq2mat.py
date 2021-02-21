import torch
import torch.nn as nn
import torch.jit as jit 

class CatReduce(jit.ScriptModule):
    
    __constants__ = ['n_in', 'n_out']
    
    def __init__(self, n_in, n_out):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        
        self.reduce = nn.Linear(n_in*2, n_out)
        
    @jit.script_method
    def forward(self, x, y):
        
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None])
        
        s = torch.cat([x, y], -1)
        
        s = self.reduce(s)
        
        return s
