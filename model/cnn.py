import pytorch_lightning as pl
import torch
import torch.nn as nn


class CNN(pl.LightningModule):    
    def __init__(self):
        super().__init__()

        self.c1 =  nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),  #32x30
            nn.ReLU(),                        
            nn.MaxPool2d(kernel_size=2)       #32x15
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  #64x12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)        #64x6
        )

        self.dense = nn.Linear(64 * 6 * 6, 10)
        self.loss = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)

        # print( x.shape )

        x = x.view( x.shape[0], -1)

        x = self.dense(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('acc', acc, prog_bar=True)
        
        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc, prog_bar=True)