import torch
from torchvision.transforms import ToTensor, Lambda, transforms
import pandas as pd
import PIL
from PIL import Image as Image
from os import listdir
import sys
sys.path.insert(1, '/content/projects-skeleton-code/train_functions')
from starting_train import initializationFunction



###Loading the dataset & Convert to Tensor
class StatementDataset(torch.utils.data.Dataset):#inherit from torch.utils.data.Dataset to make our life easier in dealing with Data
    def __init__(self, statements, labels): #image_id, labels 
        self.statements = statements  #statement are ids of the images 
        self.labels = labels 
        
    def __len__(self):
        return len(self.statements)
    def __getitem__(self, index):  #retrieve items from our dataset 
        #path needs to be changed 
        path ='/content/train_images/'
        trans1 = transforms.ToTensor()        
        statement = Image.open(path+self.statements[index]) #read specific image
        statement = trans1(statement)
        label = self.labels[index]
        return (statement,label) #return a tuple 


def StartingDataset():
    #change the path
    image_info = pd.read_csv((r'/content/train.csv'))
    image_dataset = StatementDataset(image_info.image_id,image_info.label)
    test_size = 1000
    valid_size = 1000
    train_size = len(image_dataset) - 2000
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)

    #data for initial model testing 
    train_test_size = 100
    train_rest_size = len(train_dataset) - 100
    train_test, train_rest = torch.utils.data.random_split(train_dataset, [train_test_size, train_rest_size])
    train_test_loader = torch.utils.data.DataLoader(train_test, batch_size=16, shuffle=True)

    return train_loader, valid_loader, test_loader

#initializationFunction(StartingDataset(), None)