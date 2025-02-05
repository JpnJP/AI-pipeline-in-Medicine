"""Dataset for the training of the networks."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RetinalFusionDataset(Dataset):
    def __init__(self, listImages, csv_name, transform=None):  #mean,std,
        """
        Args:

        listImages: dictionary containing the NameFiles, the processed images, the age, and the MD info
        csv_name: Path to csv file containing the image paths inside root_dir.
        transform: Required image transformation (augmentation) settings.
        """

        self.df = pd.read_csv(csv_name, index_col=0) # .csv list of patients included in the set
        self.listImages = listImages
        self.transform = transform

        self.names  = []
        self.image_list = []
        # age = []
        self.label = []

        for nameFile in self.df.index:
            # check if the file is included in the processed files
            if nameFile in list(self.listImages.keys()):
                el = self.listImages[nameFile]

                if self.transform is not None:
                    # img = self.transform(Image.fromarray((el[1][0] * 255).astype(np.uint8))) # self.
                    img = self.transform(el[0])
                else:
                    tens = torch.from_numpy((el[0] * 255).astype(np.uint8))
                    img = torch.permute(tens, (2, 0, 1))
                
                # tensors are supposed to be CxHxW, and not HxWxC (C: channel)
                self.names.append(nameFile)
                self.image_list.append(img)
                if 'DR' in csv_name:
                    if self.df.loc[nameFile, 'EyeDisorderCODE'] == 1:
                        self.label.append(1)
                    else:
                        self.label.append(0)
                elif 'Glaucoma' in csv_name:
                    if self.df.loc[nameFile, 'EyeDisorderCODE'] == 2:
                        self.label.append(1)
                    else:
                        self.label.append(0)
                else:
                    self.label.append(el[1][1])

        print("dataset size names: ", len(self.names))
        print("dataset size images: ", len(self.image_list))
        print("dataset size labels: ", len(self.label))

    def __getitem__(self, idx):
        name = self.names[idx]
        img = self.image_list[idx]
        label = torch.tensor(self.label[idx], dtype = torch.float32).unsqueeze(0) # Outupt is (1,)

        return name, img, label

    def __len__(self):
        return len(self.image_list)



