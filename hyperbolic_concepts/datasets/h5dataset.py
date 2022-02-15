import torch
import h5py

from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import h5py

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


def create_dataset(data_path: str, h5file_path: str):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor()
    ])

    animal_data = datasets.ImageFolder(data_path, transform=preprocess)
    channel, width, height = [3, 224, 224]
    N_IMAGES = (len(animal_data))
    batch_size = 512

    #TODO: Split into val, train and test.

    loader = torch.utils.data.DataLoader(animal_data,
                                              batch_size=batch_size,
                                              num_workers=2)

    with h5py.File(h5file_path, 'w') as  h5f:
      img_ds = h5f.create_dataset('images', shape=(N_IMAGES, channel, width, height), dtype='uint8', chunks=(32, 3, 224, 224))
      target_ds = h5f.create_dataset('targets', shape=(N_IMAGES), dtype=int, chunks=True)
      for i, (image, target) in enumerate(tqdm(loader)):
        img_ds[i*batch_size:(i+1)*batch_size, :, :, :] = image
        target_ds[i*batch_size:(i+1)*batch_size] = target
