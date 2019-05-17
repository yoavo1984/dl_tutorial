from torch import nn
import torch

_activation = "ReLU"

def set_activation(value):
    global _activation
    _activation = value

def get_activation():
    return ActivationLayer(_activation)
    
class ActivationLayer(nn.Module):
    def __init__(self, activation):
        super(ActivationLayer, self).__init__()
        if activation == "ReLU":
            self.activation = nn.ReLU()
        if activation == "Tanh":
            self.activation = nn.Tanh()
        if activation == "Softplus":
            self.activation = nn.Softplus()
       
    def forward(self, x):
        return self.activation(x)

class DynamicClassifier(nn.Module):
    def __init__(self, n_layers, c_start, activation="ReLU"):
        super(DynamicClassifier, self).__init__()
        set_activation(activation)
        
        self.add_module("layer_a", ResBlock(c_start, 64, 5, 2))
        for i in range(n_layers-1):
            self.add_module("layer_a_{}".format(i), ResBlock(64, 64, 5, 2))
            
        self.add_module("pool_1", nn.MaxPool2d(2))
        
        self.add_module("layer_b", ResBlock(64, 128, 5, 2))
        for i in range(n_layers):
            self.add_module("layer_b_{}".format(i), ResBlock(128, 128, 3, 1))
            
        self.add_module("pool_2", nn.MaxPool2d(2))
        
        self.add_module("layer_c", ResBlock(128, 256, 5, 2))
        for i in range(n_layers):
            self.add_module("layer_c_{}".format(i), ResBlock(256, 256, 3, 1))
        
        self.add_module("pool_3", nn.MaxPool2d(2))
        
        self.add_module("flat", Flatten())
        
        self.add_module("output", nn.Sequential(
            nn.Linear(4*4*256, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax()
        ))
    
    def forward(self, x):
        for module in self.children():
            x = module(x)
        
        return x    
        
        
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size, padding):
        super(ResBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_size, padding=(padding, padding)),
            get_activation(), 
            nn.Conv2d(out_c, out_c, k_size, padding=(padding, padding)),
            get_activation(),
            nn.BatchNorm2d(out_c)
        )
    def residual(self, x, original):
        padded = torch.zeros(x.shape)
        if next(self.parameters()).is_cuda:
            padded = padded.cuda()
            
        padded[:, 0:len(original[0])] = original
        
        return x + padded
    
    def forward(self, x):
        original = x
        x = self.block(x)
        
        # Return residual function of the input.
        return self.residual(x, original)
        

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.pool = True
        
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class CompleteConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k_size, padding, pool=True, res=False):
        super(CompleteConvLayer, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, k_size, padding=(padding, padding)), # 
            nn.ReLU(),
            nn.BatchNorm2d(out_c)
        )
        
        if pool:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = False
        
        self.res = res
        
    def forward(self, x):
        identity = x
        x = self.layer(x)
        
        if self.pool:
            x = self.pool(x)
            
        if self.res:
            x += identity
            
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        
        self.add_module("layer_1", CompleteConvLayer(3, 16, 7, 3))
        
        self.conv_1 = CompleteConvLayer(3, 16, 7, 3)
        self.res_1 = ResBlock(16, 16, 5, 2)
        # W=16, H = 16
        
        self.conv_1a = CompleteConvLayer(16, 32, 5, 2, False)
        self.res_1a = ResBlock(32, 32, 5, 2)

        
        self.conv_2 = CompleteConvLayer(32, 64, 3, 1)
        self.res_2 = ResBlock(64, 64, 3, 1)
        # W = 8, H = 8
        
        self.conv_2a = CompleteConvLayer(64, 96, 3, 1, False)
        self.res_2a = ResBlock(96, 96, 3, 1)
        
        self.conv_3 = CompleteConvLayer(96, 128, 3, 1)
        self.res_3 = ResBlock(128, 128, 3, 1)
        # W = 4, H = 4
        
        self.flat = Flatten()
        
        self.fc4 = nn.Sequential(
            nn.Linear(4*4*128, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax()
        )
        
        self.conv_layers = [
            self.conv_1, self.res_1,
            self.conv_1a, self.res_1a, 
            self.conv_2, self.res_2, 
            self.conv_2a, self.res_2a, 
            self.conv_3, self.res_3,
            self.flat]
        
        self.output = self.fc4
        
    def forward(self,x):
        for layer in self.conv_layers:
            x = layer(x)
                
        return self.output(x)
    
    def get_accuracy(self, dataset):
        correct = 0
        self.eval()
        
        for batch in dataset:
            features = batch[0]
            labels = batch[1]

            if next(self.parameters()).is_cuda:
                features = features.cuda()
                labels = labels.cuda()

            # Get predictions.
            prediction = self(features).argmax()
            correct += (labels == prediction).sum().item()
            
        self.train()
        return correct / len(dataset)

def get_model_accuracy(model, dataset):
    correct = 0
    for batch in dataset:
        features = batch[0]
        labels = batch[1]
        
        if next(model.parameters()).is_cuda:
            features = features.cuda()
            labels = labels.cuda()
        
        # Set Model to Eval mode.
#         model.eval()
        
#         import pdb
#         pdb.set_trace()
        
        # Get predictions.
        prediction = model(features).argmax(1)
        correct += (labels == prediction).sum().item()
        
    return correct / len(dataset.dataset)

