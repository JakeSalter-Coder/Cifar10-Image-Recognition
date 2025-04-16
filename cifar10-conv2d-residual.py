#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torchvision
import torchmetrics
import lightning.pytorch as pl
from torchinfo import summary
from torchview import draw_graph
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


if (torch.cuda.is_available()):
    device = ("cuda")
else:
    device = ("cpu")
print(torch.cuda.is_available())


# In[3]:


class ResidualConv2dLayer(pl.LightningModule):
    def __init__(self,
                 size,
                 **kwargs):
        super().__init__(**kwargs)
        self.conv_layer = torch.nn.Conv2d(size,
                                          size,
                                          kernel_size=(3,3),
                                          stride=(1,1),
                                          padding=(1,1))
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout2d(p=0.2)
    def forward(self, x):
        y = x
        y = self.conv_layer(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = x + y
        return y


# In[4]:


class NeuralNetwork(pl.LightningModule):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=(-10.0,10.0),
                                                translate=(0.1,0.1),
                                                scale=(0.9,1.1),
                                                shear=(-10.0,10.0)),
            torchvision.transforms.RandomHorizontalFlip(0.5),
        ])
        self.project = torch.nn.Conv2d(3, # Input channels
                                       10, # Output channels
                                       kernel_size=(3,3), # Kernel window
                                       stride=(1,1), # Step size
                                       padding=(0,0))
        self.residual = torch.nn.Sequential(*[ResidualConv2dLayer(10) for _ in range(3)])
        self.pool1_2 = torch.nn.MaxPool2d((2,2))
        self.pool3_4 = torch.nn.MaxPool2d((2,2))
        self.pool5_6 = torch.nn.MaxPool2d((2,2))
        self.flatten = torch.nn.Flatten()
        self.output_layer = torch.nn.Linear(9000,10)
        self.output_activation = torch.nn.Softmax()
        self.accuracy = torchmetrics.classification.Accuracy(task='multiclass',
                                                             num_classes=10)
        

    def forward(self, x):
        y = x
        if self.training:
             y = self.transform(y)
        y = self.project(y)
        y = self.residual(y)
        y = self.flatten(y)
        y = self.output_layer(y)
        return y

    def predict_step(self, predict_batch, batch_idx):
        x, _ = predict_batch
        y = self(x)
        y = torch.softmax(y,-1)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred,y)
        acc = self.accuracy(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred,y)
        acc = self.accuracy(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss


# In[5]:


training_dataset = torchvision.datasets.CIFAR10(root='datasets',download=True, train=True)


# In[6]:


x_train = training_dataset.data.transpose((0,3,1,2))
x_train = (x_train / 127.5) - 1.0
y_train = np.array(training_dataset.targets)
x_train.shape


# In[7]:


split_point = int(x_train.shape[0] * 0.8)
permutation = np.random.permutation(x_train.shape[0])
xy_train = torch.utils.data.DataLoader(list(zip(torch.tensor(x_train[permutation[:split_point]]).to(torch.float32),
                                                torch.tensor(y_train[permutation[:split_point]]).to(torch.long))),
                                       shuffle=True, batch_size=128,num_workers=4)
xy_val = torch.utils.data.DataLoader(list(zip(torch.tensor(x_train[permutation[split_point:]]).to(torch.float32),
                                              torch.tensor(y_train[permutation[split_point:]]).to(torch.long))),
                                       shuffle=False, batch_size=128,num_workers=4)


# In[8]:


model = NeuralNetwork()
print(model)


# In[9]:


summary(model, input_size=x_train[0:5].shape)


# In[10]:


model_graph = draw_graph(model, input_size=x_train[:split_point].shape,
                         device=device, depth=3)
model_graph.visual_graph


# In[11]:


logger = pl.loggers.CSVLogger("logs",name="cifar10_net",
                              version="conv2d-residual")
trainer = pl.Trainer(max_epochs=50,logger=logger,
                     enable_progress_bar=True,
                     log_every_n_steps=0,
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=20)])


# In[12]:


result = trainer.validate(model, dataloaders=xy_val)


# In[13]:


result = trainer.fit(model,
                     train_dataloaders=xy_train, 
                     val_dataloaders=xy_val)


# In[14]:


result = trainer.validate(model, dataloaders=xy_val)


# In[ ]:




