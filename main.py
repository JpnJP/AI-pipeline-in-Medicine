# Script for the evaluation of the PreProcessing impact on the clinical tasks related to the Retinal Eye
#In[1]: Libraries

import numpy as np
import os
import gc
from PIL import Image
import pandas as pd
import timm
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.optim import lr_scheduler
import copy
import time
import pickle

# from RetinalFusionDataset import RetinalFusionDataset
from RetinalFusionDataset import RetinalFusionDataset
from Network import networkArchitecture, set_model_gpu_mode, train_model, lossFunction, optimizer


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

#In[2]:

def main():
    # clinical aim
    clinicalAims = ['AgePrection', 'DR', 'Glaucoma', ...]

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataroot = '/home1/jovanapanic/PreProcessing_Impact/data/preprocessed_images/' # 'preprocessed/SN/' # All IMGS
    dataroot = 'TrainJP/data/'
    runTurns = ['FIRST_run', 'SECOND_run', 'THIRD_run']
    epochs = 100
    datasets = ['TR', 'VAL', 'TST']

    model_architectures = ['resnet50', 'swin_base_patch4_window12_384', ...]
    pretrained_list = ['ImageNet', 'NoPreTrain']
    dataAugmentation_list = ['NoDataAug', 'DataAug']
    lossFunctions = ['MAE', 'MSE', 'Huber', 'BinaryCross', ...]
    contrast_enhancement = ['None', 'CLAHE']  # ,

    num_workers = 8
    learning_rate = 0.001
    image_size = 384
    start_epoch = 0

    optimizerDef = 'Adam'
    resume_path = None
    batch_size = 128
    evaluation_batch_size = 128

    for clinicalAim in clinicalAims:
        # Paths were the .csv files containing the images used for each set
        val_csv = 'VAL_'+clinicalAim+'.csv'
        train_csv = 'TR_'+clinicalAim+'.csv'
        test_csv = 'TST_'+clinicalAim+'.csv'

        saving_path = 'MODELS_'+clinicalAim

        for preProcess in contrast_enhancement:
            print('Loading the data...\n')
            processedFILES = pickle.load(open(preProcess + '.dat', 'rb'))
            print('Loaded the data file.\n')

            for pretrained in pretrained_list:
                for dataAugmentation in dataAugmentation_list:

                    transform = transforms.Compose([
                                transforms.Resize([image_size,image_size]),
                                transforms.ToTensor()
                            ])

                    print("Initializing Datasets and Dataloaders...")
                    if dataAugmentation == 'DataAug':
                        dataTrain = RetinalFusionDataset(
                                    listImages = processedFILES,
                                    csv_name = train_csv,
                                    transform = transforms.Compose([
                                                                transforms.Resize([image_size,image_size]),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomVerticalFlip(),
                                                                transforms.ToTensor(),
                                                                ]),
                                    )
                    else:
                        dataTrain = RetinalFusionDataset(
                                    listImages = processedFILES,
                                    csv_name = train_csv,
                                    transform = transform
                                )

                    dataVal = RetinalFusionDataset(
                                listImages = processedFILES,
                                csv_name = val_csv,
                                transform = transform
                                )

                    dataTest = RetinalFusionDataset(
                                listImages = processedFILES,
                                csv_name = test_csv,
                                transform = transform
                                )

                    # Re-instantiate training dataloader to generate a triplet list for this training epoch
                    train_dataloader={'train':torch.utils.data.DataLoader(
                                            dataset=dataTrain,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=True),
                                        'val':torch.utils.data.DataLoader(
                                            dataset=dataVal,
                                            batch_size=evaluation_batch_size,
                                            num_workers=num_workers,
                                            shuffle=False
                                        )}

                    all_dataloader = {'TR':torch.utils.data.DataLoader(
                                            dataset=dataTrain,
                                            batch_size=1,
                                            num_workers=num_workers,
                                            shuffle=True),
                                    'VAL':torch.utils.data.DataLoader(
                                            dataset=dataVal,
                                            batch_size=1,
                                            num_workers=num_workers,
                                            shuffle=False),
                                    'TST':torch.utils.data.DataLoader(
                                            dataset=dataTest,
                                            batch_size=1,
                                            num_workers=num_workers,
                                            shuffle=False)
                    }


                    for model_architecture in model_architectures:  # model_architecture = model_architectures[0]
                        for loss_Function in lossFunctions:  # loss_Function = lossFunctions[0]
                            for runTurn in runTurns: # runTurn = runTurns[0]

                                path_save = (saving_path + '/' + preProcess +'/'+model_architecture+'/'
                                            +loss_Function+'/'+pretrained+'/'+dataAugmentation+'/'+runTurn)

                                if not os.path.isdir(path_save):
                                    os.makedirs(path_save, exist_ok=True)

                                if not os.path.isfile(path_save+'/model_trained.pt'):
                                    # There is no trained network with this combination

                                    print("Starting training of the network...")

                                    # Definition of the network structure
                                    model = networkArchitecture(model_architecture, pretrained, True)
                                    model_ft, flag_train_multi_gpu = set_model_gpu_mode(model)

                                    # garbage collector
                                    gc.collect()

                                    # In[8]:

                                    # Resume from a model checkpoint
                                    if resume_path:
                                        if os.path.isfile(resume_path):
                                            print("Loading checkpoint {} ...".format(resume_path))
                                            checkpoint = torch.load(resume_path)
                                            start_epoch = checkpoint['epoch'] + 1

                                            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
                                            if flag_train_multi_gpu:
                                                model_ft.module.load_state_dict(checkpoint['model_state_dict'])
                                            else:
                                                model_ft.load_state_dict(checkpoint['model_state_dict'])
                                            print("Checkpoint loaded: start epoch from checkpoint = {}".format(start_epoch))
                                            del checkpoint
                                        else:
                                            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))
                                    else:
                                        print('none')

                                    # In[10]:
                                    print('Start Training...\n')

                                    criterion = lossFunction(loss_Function)

                                    optimizer_ft = optimizer(model, optimizerDef, learning_rate)


                                    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

                                    model_ft = train_model(model_ft,
                                                        device,
                                                        train_dataloader,
                                                        criterion,
                                                        optimizer_ft,
                                                        exp_lr_scheduler,
                                                        flag_train_multi_gpu,
                                                        path_save,
                                                        start_epoch,
                                                        epochs) # accelerator,


                                    state = {
                                            'epoch': epochs,
                                            'model_state_dict': model_ft.state_dict(),
                                            'model_architecture': model_architecture,
                                            }

                                    # For storing data parallel model's state dictionary without 'module' parameter
                                    if flag_train_multi_gpu:
                                        state['model_state_dict'] = model_ft.module.state_dict()

                                    # Save model checkpoint
                                    torch.save(state, path_save + '/model_trained.pt'.format(
                                                    model_architecture,
                                                    epochs)
                                                    )

                                    # del train_dataloader, state
                                    torch.cuda.empty_cache()
                                    gc.collect()

                                    print('Finito training, ora applico il modello per vedere come va.\n')

                                    #In[3]:
                                    try:
                                        # Application of the model trained on the three datasets
                                        model_ft.eval()

                                        for dataset in datasets:
                                            # dataset = datasets[0]
                                            print('Start of ' + dataset)

                                            # Iterate over data.
                                            progress_bar = enumerate(tqdm(all_dataloader[dataset]))

                                            predictions = []
                                            # Iterate over data.

                                            # for batch_index, (inputs, labels) in progress_bar:
                                            for batch_index, (name, inputs, labels) in progress_bar:
                                                # batch_index, (name, inputs, labels) = next(progress_bar)
                                                inputs = inputs.to(device)
                                                # labels = labels.to(device)
                                                outcome = ((model_ft(inputs)).item())
                                                predictions.append([name, outcome, labels.item()])

                                            # predictions = Model_application(model_ft, dataloaderEVAL, device)
                                            # predictions = (predictions.cpu().tolist())
                                            dfResults = pd.DataFrame()
                                            row = 0

                                            for i, pred in enumerate(predictions):
                                                # i=0
                                                dfResults.loc[row, 'NameFile'] = pred[0]
                                                dfResults.loc[row, 'Age'] = pred[2]
                                                dfResults.loc[row, 'Estimation'] = pred[1]*100
                                                row+=1

                                            dfResults.to_csv(path_save+'/ResultsModel_'+dataset+'.csv', header=True, index=False)
                                            print('End del '+dataset)

                                    except:
                                        print('Could not apply the model correctly.\n')

                    del train_dataloader
    print('End.')

#In[4]:
if __name__ == '__main__':
    main()
    



