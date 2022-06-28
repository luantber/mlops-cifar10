# mlops-cifar10

This project is an implementation to solve the Kaggle's Cifar 10 challenge ( https://www.kaggle.com/competitions/cifar-10 ).


![image](https://user-images.githubusercontent.com/15067649/176201110-723dd6b7-5894-4436-bbb7-505e18e5cab3.png)


The model was implemented using a CNN Model using the pytorh library.

```python
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
```

There's a demo available on a Hugging Face Space: https://huggingface.co/spaces/luantber/cifar-10

![image](https://user-images.githubusercontent.com/15067649/176201686-b0a8729d-4b14-4126-b424-f7db8bca4bc6.png)

The model was deployed using a github Action, to the Hugging Face repository. See `.github/workflows`
