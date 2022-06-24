import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

translation = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
inverted_translation = {v: k for k, v in translation.items()}


class CifarDataset(Dataset):
    def __init__(self, file, folder, include_idx=False):
        self.data = pd.read_csv(file)
        self.data['label'] = self.data['label'].map(translation)
        self.folder = folder
        self.include_idx = include_idx

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        id, label = self.data.iloc[idx][0] , self.data.iloc[idx][1]
        image = read_image( f'{self.folder}/{id}.png')
        image = image/255.0
        
        if self.include_idx:
            return image, label, id
        else:
            return image, label

if __name__ == '__main__':
    dataset = CifarDataset('trainLabels.csv', 'train')
    print(dataset[0][0].type())