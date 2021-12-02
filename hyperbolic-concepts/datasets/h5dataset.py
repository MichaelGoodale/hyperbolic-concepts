import torch
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
