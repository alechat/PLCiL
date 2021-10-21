import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from models.IncrementalNetwork import IncrementalNet
from utils.accuracy import test


def losses_pseudolabeling(n_known, logits, weak_values, weak_labels, old_weak_values, old_weak_labels, tau):
    C_old = n_known
    C_new = logits.size(1)
    alpha = C_old/C_new
    
    selfsup_loss = F.cross_entropy(logits, weak_labels, reduction='none')
    selfsup_loss = selfsup_loss[weak_values >= tau].sum() / logits.size(0)
    
    old_selfsup_loss = F.cross_entropy(logits, old_weak_labels, reduction='none')
    old_selfsup_loss = old_selfsup_loss[old_weak_values >= tau].sum() / logits.size(0)
    
    return selfsup_loss + alpha * old_selfsup_loss

def compute_loss_PLCiL(n_known, old_model, unlab_images, logits, weak_values, weak_labels, tau):
    if n_known == 0:
        loss = F.cross_entropy(logits, weak_labels, reduction='none')
        loss = loss[weak_values >= tau].sum() / logits.size(0)
    else:
        with torch.no_grad():
            old_weak_preds = old_model(unlab_images).detach()
            old_weak_probas = torch.softmax(old_weak_preds, dim=1)
            old_weak_values, old_weak_labels = old_weak_probas.max(1)
        loss = losses_pseudolabeling(n_known, logits, weak_values, weak_labels, old_weak_values, old_weak_labels, tau)
    return loss


    
class PLCiLTrainer(nn.Module):
    def __init__(self, feature_extractor, feature_size=128, device='cuda'):
        super(PLCiLTrainer, self).__init__()
        self.network = IncrementalNet(feature_extractor, feature_size)
        self.feature_size = feature_size
        self.device = device
        
        # Variables monitoring the number of classes known
        self.n_classes = 0
        self.n_known = 0
        self.old_model = None
        
        # Episodic memory
        self.exemplar_sets = []
    
    @autocast()
    def forward(self, x):
        logits = self.network(x)
        return logits
        
    def increment_classes(self, n):
        self.network._add_classes(n, self.device)
        self.n_classes += n
        
    ''' EXEMPLAR MANAGEMENT '''
    def construct_exemplar_set(self, images, m):
        # Random selection of exemplars
        idx_exemplar = list(range(len(images)))
        random.shuffle(idx_exemplar)
        idx_exemplar = idx_exemplar[:m]
        # Update the memory
        self.exemplar_sets.append(np.array(images[idx_exemplar]))
    
    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
            
    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels)
            
    def build_dataset_with_exemplars(self, dataset):
        dataset.data = np.array([])
        dataset.labels = []
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels)
            
    def save_old_model(self):
        self.old_model = self.network.copy().freeze()
            
def training_session(trainer, session_id, num_epochs, num_steps_per_epoch, optimizer, scheduler, train_loader_lab, train_loader_unlab, test_loader, 
                     tau=0.8, lambd=1., logger=None, topk=(1,), CTA=True, device='cuda'):
    data_seen = 0
    correct = 0
    
    scaler = GradScaler()
    for epoch in range(1, 1+num_epochs):
        trainer.train()
        data_seen = 0
        unlab_seen = 0
        unlab_iterator = iter(train_loader_unlab)
        prog_bar = tqdm(range(num_steps_per_epoch))
        lab_iterator = enumerate(train_loader_lab)
            
        correct = 0
        avg_loss = []
        for i in prog_bar:
            x_unlab_weak, x_unlab_strong, _ = next(unlab_iterator)
            batch_idx, (x_lab_weak, x_lab_strong, labels, policies_lab) = next(lab_iterator)
            data_seen += len(labels)
            unlab_seen += len(x_unlab_weak)
            
            x_unlab_weak, x_unlab_strong = x_unlab_weak.to(device), x_unlab_strong.to(device)
            x_lab_weak, x_lab_strong, labels = x_lab_weak.to(device), x_lab_strong.to(device), labels.long().to(device)
            
            trainer.zero_grad()
            
            with torch.no_grad():
                weak_preds = trainer(x_unlab_weak).detach()
                weak_probas = torch.softmax(weak_preds, dim=1)
                weak_values, weak_labels = weak_probas.max(1)
            
            with autocast():
                y_lab = trainer(x_lab_weak)
                sup_loss = F.cross_entropy(y_lab, labels, reduction='mean')
            
                y_unlab = trainer(x_unlab_strong)
                selfsup_loss = compute_loss_PLCiL(trainer.n_known, trainer.old_model, x_unlab_weak, y_unlab, weak_values, weak_labels, tau)
                loss = sup_loss + lambd * selfsup_loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pred = y_lab.max(1)[1]
            correct_batch = pred.eq(labels).cpu().sum().item()
            correct += correct_batch
            acc = 100.*correct/data_seen
            avg_loss.append(loss.item())
            
            if CTA:
                with torch.no_grad():
                    y_pred = trainer(x_lab_strong).detach()
                    y_probas = torch.softmax(y_pred, dim=1)
                    for y_proba, t, policy in zip(y_probas, labels, policies_lab):
                        error = y_proba
                        error[t] -= 1
                        error = torch.abs(error).mean()
                        train_loader_unlab.dataset.cta.update_rates(policy, 1.0 - 0.5 * error.item())
                
            prog_bar.set_description('Session {} Epoch: {} - loss: {:.5f} - sup: {:.5f} - self: {:.5f} - Acc: {:.2f}'.format(
                                        session_id, epoch, loss.item(), sup_loss.item(), selfsup_loss.item(), acc))
        
        if logger:
            logger.info('Session {} Epoch: {} - average loss: {:.5f} - Training acc: {:.2f} - Current lr: {:.5f}'.format(
                                            session_id, epoch, np.mean(avg_loss), acc, scheduler.get_last_lr()[0]))
        
        if epoch%10 == 0:
            trainer.eval()
            with torch.no_grad():
                val_acc, correct = test(trainer, test_loader, nb_classes=trainer.n_classes, topk=topk, device=device)
                if logger:
                    if len(val_acc) == 1:
                        logger.info('Validation accuracy: {:.2f} [{}/{}]'.format(val_acc[0], correct[0], len(test_loader.dataset)))
                    else:
                        logger.info('Top1 accuracy: {:.2f} [{}/{}]'.format(val_acc[0], correct[0], len(test_loader.dataset)))
                        logger.info('Top5 accuracy: {:.2f} [{}/{}]'.format(val_acc[1], correct[1], len(test_loader.dataset)))
                        
