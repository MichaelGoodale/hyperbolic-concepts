import torch
import h5py

from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import h5py

from typing import List

class H5Dataset(torch.utils.data.Dataset):

  def __init__(self, file_path, target_transform=lambda x: x, transform=lambda x:x):
      super(H5Dataset, self).__init__()
      self.file_path = file_path
      with h5py.File(file_path, 'r') as f:
        self.length = f['targets'].shape[0]
      self.target_transform = target_transform
      self.transform = transform
      
  def __getitem__(self, index):
    if not hasattr(self, 'h5_file'):
      self.h5_file = h5py.File(self.file_path , 'r')
      self.images = self.h5_file['images']
      self.targets = self.h5_file['targets']
    return (self.transform(torch.from_numpy(self.images[index])),
            self.target_transform(self.targets[index]))
    
  def __del__(self):
      if hasattr(self, 'h5_file'):
          self.h5_file.close()

  def __len__(self):
      return self.length


def create_dataset(data_path: str, h5file_path: str) -> List[str]:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor()
    ])

    animal_data = datasets.ImageFolder(data_path, transform=preprocess)
    channel, width, height = [3, 224, 224]
    batch_size = 512

    train_length = int(len(animal_data)*0.7)
    test_length = int(len(animal_data)*0.2)
    val_length = len(animal_data) - test_length - train_length
    splits = torch.utils.data.random_split(animal_data, [train_length, test_length, val_length])
    names = [h5file_path+name+'.hdf5' for name in ['train', 'test', 'val']]

    for name, split in zip(names, splits):
        loader = torch.utils.data.DataLoader(split,
                                                  batch_size=batch_size,
                                                  num_workers=2)
        with h5py.File(name, 'w') as  h5f:
          img_ds = h5f.create_dataset('images', shape=(len(split), channel, width, height), dtype='uint8', chunks=(32, 3, 224, 224))
          target_ds = h5f.create_dataset('targets', shape=(len(split)), dtype=int, chunks=True)
          for i, (image, target) in enumerate(tqdm(loader)):
            img_ds[i*batch_size:(i+1)*batch_size, :, :, :] = image
            target_ds[i*batch_size:(i+1)*batch_size] = target
    return names 
