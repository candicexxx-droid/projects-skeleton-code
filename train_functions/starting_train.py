import torch
import torch.nn as nn
import torch.optim as optim
import wandb
# from torch.utils.tensorboard import SummaryWriter
# import sys
# sys.path.insert(1, '/content/projects-skeleton-code/networks')
# from StartingNetwork import CNN
import time
#####contains:
#find_acc
#train
#validate


def find_acc(pred, label): #calculate accuracy for each batch 
    """pixelwise accuracy"""
    correct = pred.argmax(dim = 1).eq(label)
    accuracy = correct.to(torch.float32).mean().item() * 100 
    correct_label=correct.to(torch.float32).sum()
    #
    return correct_label,accuracy

def train(network, epoch, criterion, optimizer, trainloader):
    loss_train = 0
    cor_train = 0
    acc_train = 0
    network.train()
    #epoch_no=epoch_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device at starting_train: ', device)
    for step in range(len(trainloader)): #go thru each images in one epoch

        images , labels = next(iter(trainloader))
        #images , labels = images.to(device) , labels.to(device) 
        #print("Images: " , images.shape)
        #
        # move the images and labels to GPU
        images = images.to(device)
        labels = labels.to(device)
        
        pred = network(images)
        #print('pred_train shape: ', pred.shape)
        
        # clear all the gradients before calculating them
        optimizer.zero_grad()
        
        # find the loss for the current step
        loss_train_step = criterion(pred , labels)
        
        # find accuracy
        cor_train_step, acc_train_step = find_acc(pred, labels) 
        #cor_train_step: get the total number of labels being correctly predicted at each bath 
        #acc_train_step: calculating accuracy for 16 image prediction 
        #print('acc_train_step:', acc_train_step)
        # calculate the gradients
        loss_train_step.backward()
        
        # update the parameters
        optimizer.step()
        
        loss_train += loss_train_step.item()
        cor_train += cor_train_step  
        acc_train += acc_train_step  

        if step % 5 ==0: #log the loss and accuracy every 5 steps
          wandb.log({
          #"Step {}".format(epoch_no):step,
          "Train Loss": loss_train_step, #log the loss for each train step 
          "Train Acc": cor_train/((step+1)*len(images)), #calculate the accuracy = number of correct labels so far/number of imges being passed into the model 
          # "Valid Loss": loss_val,
          # "Valid Acc": acc_valid
          })
          print('current acc:', cor_train/((step+1)*len(images)))
            
    loss_train /= (len(trainloader)) 
        #number of images in each train loader * total train loader ??no

        #not sure if here the length should be train loader(original was test loader)
    acc_train /= (len(trainloader)) #calculates the avg accuracy over all batches in each epoch 

    return loss_train, acc_train  
        
def validate(network, epoch, criterion, optimizer, testloader): 
    loss_valid = 0
    acc_valid = 0  
    cor_valid = 0     
    network.eval()  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for step in range(len(testloader)):

        images , labels = next(iter(testloader))
        
        # move the images and labels to GPU
        images = images.to(device)
        labels = labels.to(device)
        
        pred = network(images)
        
        # clear all the gradients before calculating them
        optimizer.zero_grad()
        
        # find the loss and acc for the current step
        loss_valid_step = criterion(pred , labels)
        cor_valid_step, acc_valid_step = find_acc(pred, labels)
      
        loss_valid += loss_valid_step.item()
        cor_valid += cor_valid_step  
        acc_valid += acc_valid_step

        if step % 5 ==0: #log the loss and accuracy every 5 steps
          wandb.log({
          #"Step {}".format(epoch_no):step,
          # "Train Loss": loss_train,
          # "Train Acc": acc_train,
          "Valid Loss": loss_valid,
          "Valid Acc": cor_valid/((step+1)*len(images))})

        #not sure if here the length should be testloader(original was trainloader)
    loss_valid /= (len(testloader))
    acc_valid /= (len(testloader))

    return loss_valid, acc_valid
# network, epoch, criterion, optimizer, trainloader, deivce
def starting_train(train_loader, valid_loader, training_date, test, network, num_epochs = 100):
  if test: 
    trainset_size = 3000
    
  else: 
    trainset_size = 'complete'
    
  wandb.init(name='ACM-Project_ResNet with batchnorm on Trainsize = {} ({})'.format(trainset_size, training_date), 
            project='test_project',
            #  notes='5th run', 
            tags=['Cassava-Leaf', 'Test Run'],
            entity='candicecai26')
  
  # WandB Configurations (optional)        
  wandb.config.lr = 0.01    

  #network = CNN().to(device) #model initialization 
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(network.parameters(), wandb.config.lr)

  wandb.watch(network) # Log the network weight histograms (optional)

  num_epochs = 100
  start_time = time.time()
  for epoch in range(1, num_epochs+1):
    
    #helper function 1
    loss_train, acc_train = train(network, epoch, criterion, optimizer, train_loader)
    
    #Helper function 2
    #do not evalulate the model per epoch 
    #Evaluate and save every 3 epoch 
    if epoch % 2 ==0: 
      loss_valid, acc_valid = validate(network, epoch, criterion, optimizer, valid_loader)
      print('Epoch: {}  Train Loss: {:.4f}  Train Acc: {:.4f}  Valid Loss: {:.4f}  Valid Acc: {:.4f}'.format(epoch, loss_train, acc_train, loss_valid, acc_valid))
      wandb.log({
        "Epoch": epoch,
        "Epoch Valid Loss": loss_valid,
        "Epoch Valid Acc": acc_valid})
      torch.save(network.state_dict(),'/content/drive/MyDrive/ACM AI/ResnetModel_{}（Epoch No.{}).pt'.format(training_date, epoch))
    
    print('Epoch: {}  Train Loss: {:.4f}  Train Acc: {:.4f} '.format(epoch, loss_train, acc_train))

    # Log the loss and accuracy values at the end of each epoch
    wandb.log({
        "Epoch": epoch,
        "Epoch Train Loss": loss_train,
        "Epoch Train Acc": acc_train
        # "Epoch Valid Loss": loss_valid,
        # "Epoch Valid Acc": acc_valid
        })
    
    training_time = time.time() - start_time
    print("Time Elapsed : {:.4f}s".format(training_time))

