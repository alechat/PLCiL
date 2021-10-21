import copy
import torch.nn as nn
from torch.cuda.amp import autocast

class IncrementalNet(nn.Module):
    def __init__(self, feature_extractor, feature_size, use_bias=False):
        super(IncrementalNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_size = feature_size
        
        self.classifier = None
        self.n_classes = 0
        self.use_bias = use_bias
        self.bias_layer = None
        
    @autocast()
    def forward(self, x, get_feats=False):
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        if get_feats:
            return logits, x
        return logits
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)
    
    def _add_classes(self, n_classes, device='cuda'):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)
        
        classifier = self._gen_classifier(self.n_classes + n_classes)

        if self.classifier is not None:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        self.classifier = classifier.to(device)
        self.n_classes += n_classes

    def _gen_classifier(self, n_classes):
        classifier = nn.Linear(self.feature_size, n_classes, bias=self.use_bias)
        nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.)
        return classifier