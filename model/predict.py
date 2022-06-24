from model.cnn import CNN
import torch
from dataset.dataset import inverted_translation
import matplotlib.pyplot as plt 
from torchvision.transforms import functional as F

path = "model/epoch=599-step=187800.ckpt"
model = CNN.load_from_checkpoint(path)
model.eval()

def predict(image, get_dictionary=False):
    
    image_tensor = image.view(1, 3, 32, 32)
    result = model(image_tensor)
    result = torch.softmax(result,dim=1)
    result = result[0]

    if get_dictionary:
        dict_results = {}
        
        for i in range(len(result)):
            dict_results[inverted_translation[i]] = float(result[i])

        return dict_results

    else:
        best = int(torch.argmax(result))        
        return inverted_translation[best]


