import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class MammographyDataset(Dataset):
    def __init__(self, image_dir, file_list_cc, file_list_mlo, transform=None):
        self.image_dir = image_dir
        self.file_name_cc, self.label_cc = self.__load_label_list(file_list_cc)
        self.file_name_mlo, self.label_mlo = self.__load_label_list(file_list_mlo)
        self.transform = transform

    def __load_label_list(self, file_list):
        labels, file_names_list = [], []
        with open(file_list, 'r') as f:
            for line in f:
                items = line.split()
                image_name, label = items[0], items[1:]
                label = [int(i) for i in label]
                file_names_list.append(image_name)
                labels.append(label)
        return file_names_list, labels

    def __getitem__(self, index):
        image_name_cc = self.file_name_cc[index]
        image_name_mlo = self.file_name_mlo[index]
        image_cc = Image.open(os.path.join(self.image_dir+'/CC', image_name_cc)).convert('RGB')
        image_mlo = Image.open(os.path.join(self.image_dir+'/MLO', image_name_mlo)).convert('RGB')
        label_cc = self.label_cc[index]
        label_mlo = self.label_mlo[index]

        if self.transform is not None:
            image_cc = self.transform(image_cc)
            image_mlo = self.transform(image_mlo)
        return image_cc, image_mlo, image_name_cc, image_name_mlo, \
               list(label_cc/np.sum(label_cc)), list(label_mlo/np.sum(label_mlo))

    def __len__(self):
        return len(self.file_name_cc)

def collate_fn(data):
    images_cc, images_mlo, image_id_cc, image_id_mlo, label_cc, label_mlo = zip(*data)
    images_cc = torch.stack(images_cc, 0)
    images_mlo = torch.stack(images_mlo, 0)
    return images_cc, images_mlo, image_id_cc, image_id_mlo, torch.Tensor(label_cc), torch.Tensor(label_mlo)

def get_loader(image_dir, file_list_cc, file_list_mlo, transform, batch_size, shuffle=False):
    dataset = MammographyDataset(image_dir=image_dir,
                                 file_list_cc=file_list_cc,
                                 file_list_mlo=file_list_mlo,
                                 transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader