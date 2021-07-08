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
        #all pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transform1 = transforms.Compose([normalize])
        self.transform = transform1
        
    def __len__(self):
        return len(self.statements)
    def __getitem__(self, index):  #retrieve items from our dataset 
        #path needs to be changed 
        path ='/content/train_images/'
        trans1 = transforms.ToTensor()        
        statement = Image.open(path+self.statements[index]) #read specific image
        statement = trans1(statement)
        statement = self.transform(statement)#normalize
        label = self.labels[index]
        return (statement,label) #return a tuple 


def StartingDataset():
    #change the path
    image_info = pd.read_csv((r'/content/train.csv'))
    image_dataset = StatementDataset(image_info.image_id,image_info.label)
    test_size = 1000
    valid_size = 1000
    train_size = len(image_dataset) - 2000
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size, valid_size], generator=torch.Generator().manual_seed(42))

    
)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)

    #data for initial model testing 
    train_test_size = 3000
    valid_test_size = 20
    test_test_size = 20 
    #size of the rest of the data in the training set
    train_rest_size = len(train_dataset) - train_test_size - valid_test_size - test_test_size  
    train_test, valid_test,test_test, train_rest = torch.utils.data.random_split(train_dataset, [train_test_size,valid_test_size, test_test_size, train_rest_size], generator=torch.Generator().manual_seed(42))
    train_test_loader = torch.utils.data.DataLoader(train_test, batch_size=16, shuffle=True)
    valid_test_loader = torch.utils.data.DataLoader(valid_test, batch_size=16, shuffle=True)
    test_test_loader = torch.utils.data.DataLoader(test_test, batch_size=16, shuffle=True)

    return train_test_loader, valid_loader, test_loader #fix the validation set for evaluation 


