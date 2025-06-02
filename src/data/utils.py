
import torchvision
from torch.utils.data import SequentialSampler
import h5py

class UnNormalize(torchvision.transforms.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)



class SubsetSequentialSampler(SequentialSampler):
    r'''
    Samples elements sequentially from given list of indices, without replacement.

    Args:
        indices (list): indices
    '''

    def __init__(self, indices: list):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



def print_h5py(filepath):
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            print("\t\tGroup:", key)

            group = f[key]
            for attr in group.attrs.keys():
                print("\t\t\tAttributes:", attr, group.attrs[attr])

            if isinstance(group, h5py.Group):
                for key in group.keys():
                    data = group[key]
                    print("\t\tDataset:", key, data.shape)
                    for attr in data.attrs.keys():
                        print("\t\t\tAttributes:", attr, data.attrs[attr])
        f.close()