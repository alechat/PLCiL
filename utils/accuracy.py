import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
    
def test(model, test_loader, nb_classes, topk=(1,), device='cuda', bic=False):
    model.eval()
    with torch.no_grad():
        correct = [0]*len(topk)
        for batch_idx, (X, target) in enumerate(test_loader):
            X, target = X.to(device), target.to(device)
            if bic:
                pred_ = model(X, bic=True)
            else:
                pred_ = model(X)
            _, preds = torch.max(pred_, 1)
            res = accuracy(pred_, target, topk)
            for i in range(len(res)):
                correct[i] += res[i].cpu().item()
        val_acc = [0]*len(topk)
        for i in range(len(correct)):
            val_acc[i] = 100. * float(correct[i]) / float(len(test_loader.dataset))
        return val_acc, correct