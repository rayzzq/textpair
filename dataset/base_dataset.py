import torch 


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(BaseDataset, self).__init__()
        self.data = data 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

