import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def path_loader(path):
    return path

class ImageNet(ImageFolder):
    def __init__(self, root, classes, train=True, loader=path_loader, full=False):
        super(ImageNet, self).__init__(os.path.join(root, 'train') if train else os.path.join(root, 'val'), loader=loader, transform=None, target_transform=None)
        
        self.classes_id = classes
        if full:
            removed = 0
            core_classes = np.load(os.path.join(root, 'imagenet_core_classes.npy'))
            for c in self.classes_id.copy():
                if self.classes[c] in core_classes:
                    self.classes_id.remove(c)
                    removed+= 1
            print('%d classes removed'%removed)
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

        
    
dir_train = 'Imagenet32_train_npz'
dir_val = 'Imagenet32_val_npz'

class ImageNet32(Dataset):
    def __init__(self, root,
                 classes=range(1000),
                 train=True,
                 transform=None,
                 target_transform=None,
                 permute=None):
        super(ImageNet32, self).__init__()
        
        self.train = train
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        self.permute = permute

        if permute:
            classes = permute[classes[0]:classes[0]+len(classes)]
        self.classes = classes
        if train==True:
            for i in range(1, 11):
                f = os.path.join(root, dir_train, 'train_data_batch_%d.npz'%i)
                data_dic = np.load(f)
                if i == 1:
                    data = data_dic['data']
                    labels = data_dic['labels']
                else:
                    data = np.concatenate((data, data_dic['data']))
                    labels = np.concatenate((labels, data_dic['labels']))
                    print(data.shape)
        else:
            f = os.path.join(root, dir_val, 'val_data.npz')
            data_dic = np.load(f)
            data = data_dic['data']
            labels = np.array(data_dic['labels'])
        
        data = data.reshape((-1, 3, 32, 32))
        data = np.transpose(data, (0,2,3,1))
        self.data = data
        self.targets = labels-1
        
        # Select subset of classes
        per_class_data = {i:[] for i in classes}
        for i in range(len(self.data)):
            if self.targets[i] in classes:
                per_class_data[self.targets[i]].append(self.data[i])
        
        data = []
        targets = []
        for c in classes:
            data += per_class_data[c]
            if permute:
                targets += [permute.index(c)]*len(per_class_data[c])
            else:
                targets += [c]*len(per_class_data[c])
        
        self.data = np.array(data)
        self.targets = targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target

    def __len__(self):
        return len(self.data)
    
    def get_class_idx(self, label):
        return np.where(np.array(self.targets) == label)[0]

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]
    
    def get_classes(self):
        return self.classes

    def append(self, images, labels):
        self.data = np.concatenate((self.data, images), axis=0)
        self.targets = self.targets + labels
        

