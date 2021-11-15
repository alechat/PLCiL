import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import os
import random
import numpy as np
import torch

from datasets.CTAugment import CTAugment
import datasets.prepare_datasets as datasets
from utils.sampler import RandomSampler, BatchSampler
from utils.accuracy import test

from models.PLCiL import PLCiLTrainer, training_session
from models.networks import cifar_wide_resnet, resnet


log = logging.getLogger(__name__)

@hydra.main(config_path='config/', config_name='plcil')
def main_plcil(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Select device, GPU or CPU
    device = torch.device("cuda:{}".format(cfg.gpu)  if torch.cuda.is_available() else "cpu")
    print('Main device: {}'.format(device))
    
    # Set seed
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Current working directory
    dir_output = os.getcwd()
    print('Working Directory: {}'.format(dir_output))
    
    # Permute the order in which the classes are introduced. Split the classes into different sessions with a given incremental step.
    permute = np.arange(cfg.dataset.n_classes)
    np.random.shuffle(permute)
    np.save('permute', permute)
    permute = list(permute)
    learning_order = [list(range(i,i+cfg.dataset.incremental_step)) for i in range(0, cfg.dataset.n_classes, cfg.dataset.incremental_step)]

    # Experiment parameters
    dataset_name = cfg.dataset.name
    batch_size = cfg.params.batch_size
    mu = cfg.params.mu
    K = cfg.params.memory
    
    # Initialize the incremental trainer with the selected backone
    if cfg.model == 'resnet32':
        model = resnet.CifarResnet32(num_classes=0)
        n_feats = 64
    elif cfg.model == 'wrn28-8':
        model = cifar_wide_resnet.Wide_ResNet(28, 8, 0, num_classes=0, leaky=True)
        n_feats = 512
    elif cfg.model == 'resnet18':
        model = resnet.resnet18(0)
        n_feats = 512
    
    trainer = PLCiLTrainer(model, feature_size=n_feats, device=device)
    trainer = trainer.to(device)

    
    # Initialize the unlabeled pool of data    
    idx_dic = None
    topk = (1,)
    cta = CTAugment()
    if dataset_name == 'cifar100':
        unlab_dataset = datasets.get_imagenet32(cfg.datasetpath.imagenet32, cta)
    elif dataset_name == 'imagenet100':
        topk = (1,5,)
        if cfg.dataset.unlab == 'places365':
            unlab_dataset = datasets.get_Places365_unlab(cfg.datasetpath.places365, cta)
        else:
            if cfg.dataset.unlab == 'equal':
                idx_dic = datasets.split_datasets(cfg.dataset.n_classes, 1300)
            else:
                unlab_dataset = datasets.get_ImageNet100_unlab(cfg.datasetpath.imagenet, cta, mode=cfg.dataset.unlab, idx_dic=idx_dic)
    
    
    # Incremental training process
    test_classes = []
    session_id = 0
    for s in learning_order:
        session_id += 1
        test_classes += s
        
        # Build the dataset instances for the labeled data for training and validation.
        if dataset_name == 'cifar100':
            lab_dataset = datasets.get_cifar_lab(cfg.datasetpath.cifar, s, cfg.dataset.n_lab, idx_dic, cta=cta, permute=permute)
            test_dataset = datasets.get_cifar_val(cfg.datasetpath.cifar, test_classes, permute=permute)
        elif dataset_name == 'imagenet100':
            lab_dataset = datasets.get_ImageNet_lab(cfg.datasetpath.imagenet, s, cfg.dataset.n_lab, idx_dic, cta=cta, permute=permute)
            test_dataset = datasets.get_ImageNet_val(cfg.datasetpath.imagenet, test_classes, permute=permute)
        
        # Merge new labeled data with exemplars stored in the episodic memory
        trainer.combine_dataset_with_exemplars(lab_dataset)
        
        # Compute the number of mini-batches required to see the whole labeled dataset
        num_steps_per_epoch = int(np.ceil(len(lab_dataset)/batch_size))
        
        # Sample "params.size_unlab" samples from the unlabeled data pool to build the unlabeled dataset available during the current session
        # We use RandomSampler and BatchSampler to ensure that for each labeled mini-batch of size "batch_size", 
        # we also provide the model with "mu*batch_size" unlabeled images
        unlab_subset = unlab_dataset.build_subset(cfg.params.size_unlab)
        sampler_unlab = RandomSampler(unlab_subset, replacement=True, num_samples=num_steps_per_epoch * batch_size * mu)
        batch_sampler_unlab = BatchSampler(sampler_unlab, batch_size * mu, drop_last=True)
        unlab_loader = torch.utils.data.DataLoader(unlab_subset, batch_sampler=batch_sampler_unlab, pin_memory=False, num_workers=cfg.num_workers)
        
        sampler_lab = RandomSampler(lab_dataset, replacement=True, num_samples=num_steps_per_epoch * batch_size)
        batch_sampler_lab = BatchSampler(sampler_lab, batch_size, drop_last=True)
        lab_loader = torch.utils.data.DataLoader(lab_dataset, batch_sampler=batch_sampler_lab, 
                                                 collate_fn=datasets.collate_policy, pin_memory=False,  num_workers=cfg.num_workers)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers)
        
        
        # Add new outputs to the model
        trainer.increment_classes(len(s))
        
        # Optimizer and scheduler
        optimizer = torch.optim.SGD(trainer.parameters(), lr=cfg.params.lr, momentum=.9, nesterov=True, weight_decay=cfg.params.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10*num_steps_per_epoch, T_mult=2)
    
        # Training
        training_session(trainer, session_id, cfg.params.epochs, num_steps_per_epoch, optimizer, scheduler, lab_loader, unlab_loader, test_loader,
                                 tau=cfg.params.tau, lambd=cfg.params.lamb, logger=log, topk=topk, device=device)
        
        # Save a snapshot of the model and update the episodic memory
        trainer.save_old_model()
        m = K // trainer.n_classes
        trainer.reduce_exemplar_sets(m)
        for y in range(trainer.n_known, trainer.n_classes):
            images = lab_dataset.get_image_class(y)
            trainer.construct_exemplar_set(images, m)
        trainer.n_known = trainer.n_classes
        
        # Evaluation of the perfomance at the end of the session on all the classes learned so far.
        trainer.eval()
        with torch.no_grad():
            acc, correct = test(trainer, test_loader, nb_classes=trainer.n_classes, topk=topk, device=device)
            if len(acc) == 1:
                log.info('Session {} ({} classes) DONE! Final accuracy: {:.2f} [{}/{}]'.format(session_id, trainer.n_known, acc[0], correct[0], len(test_loader.dataset)))
            else:
                log.info('Session {} ({} classes) DONE! Top1 accuracy: {:.2f} [{}/{}]'.format(session_id, trainer.n_known, acc[0], correct[0], len(test_loader.dataset)))
                log.info('Session {} ({} classes) DONE! Top5 accuracy: {:.2f} [{}/{}]'.format(session_id, trainer.n_known, acc[1], correct[1], len(test_loader.dataset)))
            
            if cfg.save_model:
                state = trainer.network.state_dict()
                torch.save(state, os.path.join(dir_output, 'Model_%02d.t7'%(session_id)))
    

if __name__ == '__main__':
    main_plcil()