import os
import numpy as np
from torchvision.datasets import ImageFolder


def path_loader(path):
    return path

class Places365(ImageFolder):
    def __init__(self, root, classes, train=True, loader=path_loader):
        super(Places365, self).__init__(os.path.join(root, 'train') if train else os.path.join(root, 'val'), loader=loader, transform=None, target_transform=None)
        
        self.classes_id = classes
        img_list = []
        targets_list = []
        for x, y in self.samples:
            if y in self.classes_id:
                img_list.append(x)
                targets_list.append(y)
        self.data = img_list
        self.targets = targets_list
        
        self.classes = list(np.array(self.classes)[self.classes_id])
        new_dic = {i:self.class_to_idx[i] for i in self.classes}
        self.class_to_idx = new_dic
        self.samples = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path = self.data[index]
        target = self.targets[index]
        sample = self.loader(path)
        return sample, target
        

