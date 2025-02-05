import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import time
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from timm.data import resolve_data_config


def networkArchitecture(model_architecture, pretrained, regressionTask = False):
    # Create the model
    if pretrained == 'ImageNet':
        model = timm.create_model(model_architecture, pretrained=True, num_classes=1) #, num_classes=100)
    else:
        model = timm.create_model(model_architecture, pretrained=False, num_classes=1)

    if regressionTask:
        # Modify the final layer for regression
        if hasattr(model, 'fc'):  # For models like resnet
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 1)
        elif hasattr(model, 'classifier'):  # For models like efficientnet
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 1)
        elif hasattr(model, 'head'):  # For models like Swin Transformer
            # in_features = model.head.in_features
            config = resolve_data_config({}, model=model)
            input_size = config['input_size'][-1]
            # model.head = nn.Linear(in_features, 1)
            model.head = nn.Sequential(
                        # timm.models.layers.SelectAdaptivePool2d(pool_type='avg', flatten=Identity())
                        nn.AdaptiveAvgPool2d(1000),        # Global average pooling
                        nn.Flatten(),                      # Flatten the pooled output
                        nn.Dropout(p=0.0, inplace=False),  # Dropout with a probability of 0.5
                        nn.Linear(int(1000*1000*input_size/32), 1)  # Linear layer for regression
                    )
        else:
            raise ValueError("Unknown model architecture or final layer attribute")

    return model

def lossFunction(loss):
    if loss == 'MAE': # This loss function is more robust to outliers than MSE.
        return nn.L1Loss()
    elif loss == 'MSE': # This loss function is sensitive to outliers and penalizes larger errors more than smaller ones.
        return nn.MSELoss()
    elif loss == 'Huber': # The Huber Loss is a combination of MSE and MAE, controlled by the hyperparameter δ\deltaδ.
        return nn.SmoothL1Loss()
    
    elif loss == 'BinaryCross':
        return nn.BCEWithLogitsLoss() #nn.BCELoss()

def optimizer(model, optimizerDef, learning_rate):
    if optimizerDef == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizerDef == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizerDef == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizerDef =='AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate)

# Definition of random number fixing
def torch_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def set_model_gpu_mode(model):
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
        # model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        # model.cuda()
        flag_train_multi_gpu = False
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu


def train_model(model, device, dataloaders, criterion, optimizer, exp_lr_scheduler,flag_train_multi_gpu,path_save, start_epoch,num_epochs=25): #, accelerator, 
    since = time.time()
   
    train_losses = []
    val_losses = []

    best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss = 0
    best_loss = float('inf')

    if flag_train_multi_gpu:
        # Initialization with Mulilt=GPU
        model = nn.DataParallel(model)

    model = model.to(device)

    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            progress_bar = enumerate(tqdm(dataloaders[phase]))
            for i, my_stuff in progress_bar: 
                _, inputs, labels = my_stuff
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'): # will enable or disable grads based on its argument mode. It can be used as a context-manager or as a function.

                    # forward pass
                    outputs =  model(inputs)
                    loss = criterion(outputs, labels)

                    # Fix Gradient Calculation
                    # Gradient Calculation should only happen during training, not validation
                    if phase == 'train': 
                        # backward + optimize only if in training phase
                        loss.backward()
                        
                        # Optimization step
                        optimizer.step()    
                        optimizer.zero_grad()
                    
                    # Gradients are not computed during validation, which can save computation time and memory

                # statistics
                running_loss += loss.item() * inputs.size(0)

                if phase == 'val':
                    corrects = torch.sum(outputs.argmax(dim=1) == labels)
                    running_corrects += corrects.item()


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                # Learning rate scheduler should be updated after each epoch of training, not after each phase
                exp_lr_scheduler.step()
                print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            else:
                val_losses.append(epoch_loss)

                # Track the Best Loss. I should save the best loss ehenever a better validation loss is found.
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Saving the best model only!
                    if flag_train_multi_gpu:
                        model_to_save = model.module # when the flag is active, the actual model weights are stored in model.module
                    else:
                        model_to_save = model
                    torch.save({
                        'model_state_dict':model_to_save.state_dict(), 
                        'epoch': epoch
                        }, path_save + '/model_bestCheckpoint.pt')

                print(f'{phase} Loss: {epoch_loss:.4f}, Accuracy: {running_corrects/len(dataloaders[phase].dataset):.4f}')


    model.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # Save Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Trend')
    plt.legend()
    plt.savefig(path_save + '/Losses.png')
    plt.close()

    return model 

