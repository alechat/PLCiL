import torch
import copy
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
import torchvision.transforms as T

from datasets.CTAugment import apply
from datasets.ImageNet import ImageNet, ImageNet32
from datasets.Places365 import Places365


mean_imagenet, std_imagenet = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


### Data Augmentation
def get_standard_transforms(img_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), is_train=True):
    if is_train:
        if img_size == 224:
            trans_size = [T.RandomResizedCrop(224)]
        else:
            trans_size = [T.RandomCrop(img_size, padding=int(img_size*0.125))]
        trans = T.Compose(trans_size + [
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    else:
        if img_size == 224:
            trans_size = [T.Resize((256, 256)), T.CenterCrop(224)]
        else:
            trans_size = []            
        trans = T.Compose(trans_size+[
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return trans


### Core classes
def collate_policy(batch):
    data = list(zip(*batch))
    weak_data = torch.stack(data[0], 0)
    strong_data = torch.stack(data[1], 0)
    lb = torch.Tensor(data[2])
    policy = list(data[3])
    return weak_data, strong_data, lb, policy

def split_datasets(num_classes, total_per_class):
    idx_dic = {}
    for c in range(num_classes):
        idx_list = [i for i in range(total_per_class)]
        random.shuffle(idx_list)
        idx_dic[c] = idx_list
    return idx_dic

class CommonDataset(Dataset):
    def __init__(self, data, labels, is_train=True, img_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), transpose=False, loader=None):
        super(CommonDataset, self).__init__()
        self.data, self.labels = np.array(data), labels
        self.is_train = is_train
        self.transpose = transpose
        self.loader = loader
        assert len(self.data) == len(self.labels)
        
        self.classes = list(set(self.labels))
        
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.trans = get_standard_transforms(img_size, mean, std, is_train)
        
    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        if self.loader:
            im = self.loader(im)
        else:
            if self.transpose:
                im = Image.fromarray(np.transpose(im, (1, 2, 0)))
            else:
                im = Image.fromarray(im)
        im = self.trans(im)
        return im, lb

    def __len__(self):
        leng = len(self.data)
        return leng
    
    def get_classes(self):
        self.classes = list(set(self.labels))
        return self.classes
    
    def get_class_idx(self, label):
        return np.array(np.where(np.array(self.labels) == label)[0])

    def get_image_class(self, label):
        idx = self.get_class_idx(label)
        return self.data[idx]
    
    def p_y(self):
        py = []
        for i in self.classes:
            n = len(self.get_class_idx(i))
            py.append(n)
        py = torch.Tensor(py).float()
        return py/torch.sum(py)
    
    def append(self, images, labels):
        if len(self.data)==0:
            self.data = images
        else:
            self.data = np.concatenate((self.data, images), axis=0)
        self.labels = self.labels + labels
        self.classes = list(set(self.labels))

class AugmentedDataset(CommonDataset):
    def __init__(self, data, labels, cta=None, is_train=True, probe=True, img_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), transpose=False, loader=None):
        super(AugmentedDataset, self).__init__(data, labels, is_train=is_train, img_size=img_size, mean=mean, std=std, transpose=transpose, loader=loader)
        self.probe = probe
        
        if is_train:
            if img_size == 224:
                trans_size = [T.RandomResizedCrop(224)]
            else:
                trans_size = [T.RandomCrop(img_size, padding=int(img_size*0.125))]
            self.cta = cta
            self.trans_weak = get_standard_transforms(img_size, mean, std, is_train)
            
            self.trans_strong_1 = T.Compose(trans_size + [
                T.RandomHorizontalFlip(p=0.5),])
            self.trans_strong_2 = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
    
            self.cutout = T.Compose([
                    T.RandomErasing(p=1., scale=(0.02, 0.25), ratio=(1,1)),
            ])
    
        else:
            self.trans = get_standard_transforms(img_size, mean, std, is_train)
    
   
    def cta_augment(self, img):
        img = self.trans_strong_1(img)
        policy = self.cta.policy(probe=self.probe)
        img = apply(img, policy)
        img = self.trans_strong_2(img)
        img = self.cutout(img)
        return img, policy
    
    def build_subset(self, size):
        idx = torch.from_numpy(np.random.choice(len(self.data), size=size, replace=True))
        data = self.data[idx]
        labels = list(np.asarray(self.labels)[idx])
        
        return AugmentedDataset(data, labels, cta=self.cta, is_train=self.is_train, probe=self.probe, 
                                img_size=self.img_size, mean=self.mean, std=self.std, 
                                transpose=self.transpose, loader=self.loader)
        
        
    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        if self.loader:
            im = self.loader(im)
        else:
            if self.transpose:
                im = Image.fromarray(np.transpose(im, (1, 2, 0)))
            else:
                im = Image.fromarray(im)
        if self.is_train:
            if self.cta is None:
                strong_data = self.trans_strong_1(im)
                strong_data = self.trans_strong_2(strong_data)
                return strong_data, lb
            weak_data = self.trans_weak(im)
            strong_data, policy = self.cta_augment(im)
            if self.probe:
                return weak_data, strong_data, lb, policy
            else: 
                return weak_data, strong_data, lb
        else:
            return self.trans(im), lb

        
        
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


### Loading tools
def split_unlab_dataset(dset, num_lab_per_class, cta, idx_dic, img_size=32, 
                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), image_folder=False):
    data = copy.deepcopy(dset.data)
    try:
        targets = dset.targets
    except:
        targets = dset.labels
    classes = set(targets)
    per_class_data = {i:[] for i in classes}
    for i in range(len(data)):
        if targets[i] in classes:
            if image_folder:
                per_class_data[targets[i]].append(data[i])
            else:
                per_class_data[targets[i]].append(np.array(data[i]))
    
    data_unlab = []
    targets_unlab = []
    for c in classes:
        if idx_dic:
            idx_unlab = idx_dic[c][num_lab_per_class:]
        else:
            idx_unlab = list(range(len(per_class_data[c])))
        idx_unlab = np.array(idx_unlab)
        idx_unlab = idx_unlab[np.where(idx_unlab<len(per_class_data[c]))]
        data_unlab += [per_class_data[c][i] for i in idx_unlab]
        targets_unlab += [c]*len(idx_unlab)
    
    loader = None
    if image_folder:
        loader = pil_loader
    return AugmentedDataset(data_unlab, targets_unlab, cta=cta, is_train=True, probe=False, img_size=img_size, mean=mean, std=std, loader=loader)


def split_lab_dataset(dset, num_lab_per_class, classes, cta=None, idx_dic=None, permute=None, img_size=32,
                      mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), transpose=False, image_folder=False, supervised_only=False):
    if permute is not None:
        classes = permute[classes[0]:classes[0]+len(classes)]
    per_class_data = {i:[] for i in classes}
    data = copy.deepcopy(dset.data)
    try:
        targets = dset.targets
    except:
        targets = dset.labels
    for i in range(len(data)):
        if targets[i] in classes:
            if image_folder:
                per_class_data[int(targets[i])].append(data[i])
            else:
                per_class_data[int(targets[i])].append(np.array(data[i]))
    
    data_lab = []
    targets_lab = []
    for c in classes:
        if idx_dic:
            idx_lab = idx_dic[c][:num_lab_per_class]
        else:
            idx_lab = list(range(len(per_class_data[c])))
            random.shuffle(idx_lab)
            idx_lab = idx_lab[:num_lab_per_class]
        idx_lab = np.array(idx_lab)
        idx_lab = idx_lab[np.where(idx_lab<len(per_class_data[c]))]
        data_lab += [per_class_data[c][i] for i in idx_lab]
        if permute is not None:
            targets_lab += [permute.index(c)]*len(idx_lab)
        else:
            targets_lab += [c]*len(idx_lab)
    
    loader = None
    if image_folder:
        loader = pil_loader
    if supervised_only:
        return CommonDataset(data_lab, targets_lab, is_train=True, img_size=img_size, mean=mean, std=std, transpose=transpose, loader=loader)
    
    return AugmentedDataset(data_lab, targets_lab, cta=cta, is_train=True, probe=True, img_size=img_size, mean=mean, std=std, transpose=transpose, loader=loader)

def get_val_dataset(dset_test, classes_test, permute=None, img_size=32, 
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), transpose=False, image_folder=False):
    if permute is not None:
        classes_test = permute[classes_test[0]:classes_test[0]+len(classes_test)]
    per_class_data_test = {i:[] for i in classes_test}
    data_test = dset_test.data
    try:
        targets_test = dset_test.targets
    except:
        targets_test = dset_test.labels
    for i in range(len(data_test)):
        if targets_test[i] in classes_test:
            if image_folder:
                per_class_data_test[int(targets_test[i])].append(data_test[i])
            else:
                per_class_data_test[int(targets_test[i])].append(np.array(data_test[i]))
    data_test = []
    targets_test = []
    for c in classes_test:
        data_test += per_class_data_test[c]
        if permute is not None:
            targets_test += [permute.index(c)]*len(per_class_data_test[c])
        else:
            targets_test += [c]*len(per_class_data_test[c])
            
    loader = None
    if image_folder:
        loader = pil_loader
    
    return AugmentedDataset(data_test, targets_test, is_train=False, img_size=img_size, mean=mean, std=std, transpose=transpose, loader=loader)


### CIFAR100 experiments
def get_cifar_unlab(root, num_lab_per_class, idx_dic, cta):
    dset = CIFAR100(root, train=True, transform=None, target_transform=None, download=True)
    dset_unlab = split_unlab_dataset(dset, num_lab_per_class, cta, idx_dic, img_size=32)
    del dset
    return dset_unlab

def get_cifar_lab(root, classes, num_lab_per_class, idx_dic, cta=None, permute=None, supervised_only=False):
    dset = CIFAR100(root, train=True, transform=None, target_transform=None, download=True)
    dset_lab = split_lab_dataset(dset, num_lab_per_class, classes, cta, idx_dic=idx_dic, permute=permute, img_size=32, supervised_only=supervised_only)
    del dset
    return dset_lab

def get_cifar_val(root, classes_test, permute=None):
    dset_test = CIFAR100(root, train=False, transform=None, target_transform=None, download=True)
    dset_val = get_val_dataset(dset_test, classes_test, permute=permute, img_size=32)
    return dset_val

def get_imagenet32(root, cta):
    dset = ImageNet32(root, classes=range(1000), train=True, transform=None, target_transform=None, permute=None)
    dset_unlab = AugmentedDataset(dset.data, dset.targets, cta=cta, is_train=True, probe=False, img_size=32)
    return dset_unlab


###ImageNet experiments
def get_ImageNet100_unlab(root, cta, unlabeled_scenario='disjoint', idx_dic=None, num_lab_per_class=130):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if unlabeled_scenario == 'disjoint':
        dset = ImageNet(root, classes=list(range(100,1000)), train=True)
    elif unlabeled_scenario =='full':
        dset = ImageNet(root, classes=list(range(0,1000)), train=True)
    elif unlabeled_scenario =='equal':
        dset = ImageNet(root, classes=list(range(0,100)), train=True)
        if idx_dic:
            dset_unlab = split_unlab_dataset(dset, num_lab_per_class, cta, idx_dic=idx_dic, img_size=224, mean=mean, std=std, image_folder=True)
            return dset_unlab
    dset_unlab = AugmentedDataset(dset.data, dset.targets, cta=cta, is_train=True, probe=False, img_size=224, mean=mean, std=std, loader=pil_loader)
    return dset_unlab

def get_ImageNet_unlab(root, num_lab_per_class, idx_dic, cta):
    dset = ImageNet(root, classes=list(range(1000)), train=True)
    dset_unlab = split_unlab_dataset(dset, num_lab_per_class, cta, idx_dic=idx_dic, img_size=224, mean=mean_imagenet, std=std_imagenet, image_folder=True)
    return dset_unlab

def get_ImageNet_lab(root, classes, num_lab_per_class, idx_dic, cta=None, permute=None, supervised_only=False):
    dset = ImageNet(root, classes=list(range(1000)), train=True)
    dset_lab = split_lab_dataset(dset, num_lab_per_class, classes, cta, idx_dic=idx_dic, permute=permute, img_size=224, mean=mean_imagenet, std=std_imagenet, image_folder=True, supervised_only=supervised_only)
    return dset_lab

def get_ImageNet_val(root, classes_test, permute=None):
    dset_test = ImageNet(root, classes=list(range(1000)), train=False)
    dset_val = get_val_dataset(dset_test, classes_test, permute=permute, img_size=224, mean=mean_imagenet, std=std_imagenet, image_folder=True)
    return dset_val

def get_Places365_unlab(root, cta):
    dset = Places365(root, classes=list(range(365)), train=True)
    dset_unlab = AugmentedDataset(dset.data, dset.targets, cta=cta, is_train=True, probe=False, img_size=224, mean=mean_imagenet, std=std_imagenet, loader=pil_loader)
    return dset_unlab