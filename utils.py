import torch
import torch.utils.data
import shutil
from model.model import *
import os


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(path):
    """

    :param path: path to checkpoint
    :return: restored models
    """
    assert os.path.exists(path), 'Checkpoint doesn\'t exist.'
    checkpoint = torch.load(path)

    return checkpoint


def get_model(pretrain=False, path=None, device='cuda'):
    assert device in ['cuda', 'cpu'], 'Undefined device.'
    assert not pretrain or (pretrain and path is not None), 'Provide path to load pretrain model'

    if pretrain:
        return load_checkpoint(path)

    ret = {
        'Fe': F_extractor().to(device),
        'F1': F_label().to(device),
        'F2': F_label().to(device),
        'Ft': F_label().to(device)
    }

    return ret


class Concat2Dataset(torch.utils.data.Dataset):
    def __init__(self, data_0, data_1):
        super(Concat2Dataset, self).__init__()

        self.data_0 = data_0
        self.data_1 = data_1

        self.len0 = len(data_0)
        self.len1 = len(data_1)

        self.total_len = self.len0 + self.len1

    def __getitem__(self, i):
        assert i < self.total_len, 'Index is out of range.'
        return self.data_0[i] if i < self.len0 else self.data_1[i-self.len0]

    def __len__(self):
        return self.total_len


class my_dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(my_dataset, self).__init__()
        self.data = [1,2,3,4,5,6]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return 6

if __name__ == '__main__':
    a = my_dataset()
    a = Concat2Dataset(a,a)
    a_loader = torch.utils.data.DataLoader(a, batch_size=1, num_workers=1, shuffle=True)
    for _ in range(10):
        x = [i for i in a_loader]
        print(x)