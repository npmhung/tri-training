import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from model.model import *
from utils import *


def train_1(device, Fe, F1, F2, Ft, train_loader, val_loader, epochs=1):
    W1 = next(F1.parameters())
    W2 = next(F2.parameters())
    # print(F.red W1*W2)

    # print([*F1.parameters()][0].data)
    opt = optim.Adam([*F1.parameters(), *F2.parameters(), *Fe.parameters(), *Ft.parameters()], lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(opt, 100, 0.9)

    lambda_view = 1e-4
    for epoch in range(epochs):
        tqdm()
        with tqdm(total=len(train_loader.dataset), desc='Training phase 1') as pbar:

            for idx, (x, y) in enumerate(train_loader):
                scheduler.step()
                # print(y)
                # break
                x, y = x.to(device), y.to(device)
                features = Fe(x)
                f1_out = F1(features)
                f2_out = F2(features)
                ft_out = Ft(features)  # should test to use new feature Fe(x) instead of old features
                # print(features)
                view_loss = lambda_view * torch.sum(torch.abs(W1 * W2))
                loss = F.cross_entropy(f1_out, y) + F.cross_entropy(f2_out, y) + view_loss + F.cross_entropy(ft_out, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                acc = (Ft(Fe(x)).max(1)[1] == y).sum().item() / train_loader.batch_size

                tmp_1 = []
                tmp_2 = []
                tmp_t = []
                if idx % 500 == 0:
                    with tqdm(total=len(val_loader.dataset), desc='Calculate validation accuracy') as pval:
                        for idv, (xv, yv) in enumerate(val_loader):
                            xv = xv.to(device)
                            yv = yv.to(device)
                            tfeatures = Fe(xv)
                            tmp_t.append((Ft(tfeatures).max(1)[1] == yv))
                            tmp_1.append((F1(tfeatures).max(1)[1] == yv))
                            tmp_2.append((F2(tfeatures).max(1)[1] == yv))
                            pval.update(val_loader.batch_size)
                    val_acc_t = torch.mean(torch.cat(tmp_t).float())
                    val_acc_1 = torch.mean(torch.cat(tmp_1).float())
                    val_acc_2 = torch.mean(torch.cat(tmp_2).float())

                pbar.set_postfix_str('Target net Acc %f - Net 1 Acc %f - Net 2 Acc %f'%(val_acc_t, val_acc_1, val_acc_2))
                pbar.update(train_loader.batch_size)

    save_checkpoint({
        'epoch': epochs + 1,
        'Fe': Fe.state_dict(),
        'F1': F1.state_dict(),
        'F2': F2.state_dict(),
        'Ft': Ft.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict()
    }, False, 'phase1.pth.tar')


def train_2(device, Fe, F1, F2, Ft, S, T, val_data, step=50, iters = 2):
    """

    :param device:
    :param Fe:
    :param F1:
    :param F2:
    :param Ft:
    :param S: tuple(imgs, labels)
    :param T: imgs
    :param val_data:
    :param step:
    :param iters:
    :return:
    """
    k = 0

    # label
    T_loader = torch.utils.data.DataLoader(T,
                                           batch_size=128,
                                           num_workers=1)
    thres = 0.1
    while k < step:
        # label step
        T_new = []
        with tqdm(total=len(T_loader.dataset), desc='Labeling target data.') as pbar:
            for idx, (x,_) in enumerate(T_loader):
                """
                    assume that we don't have the target dataset's label 
                """
                # print(len(x))
                x = x.to(device)

                y_1 = F1(Fe(x)).max(1)
                y_2 = F2(Fe(x)).max(1)

                cond = (y_1[1]==y_2[1])
                max_prob = torch.stack([y_1[0], y_2[0]], 1).max(1)[0]*cond.float()

                _tmp = [(newx, newy) for (newx, newy, prob) in zip(x, y_1[1], max_prob) if prob > thres]
                T_new.extend(_tmp)
                pbar.update(T_loader.batch_size)

        try:
            T_new, T_new_l = zip(*T_new)
        except:
            T_new, T_new_l = [], []





        # train step
        for j in range(iters):
            pass

        # pass

def main():
    transform_mnist = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    transform_svhn = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    mnist = tv.datasets.MNIST('./dataset/mnist', download=True, transform=transform_mnist)
    mnist_val = tv.datasets.MNIST('./dataset/mnist', download=True, transform=transform_mnist, train=False)
    svhn = tv.datasets.SVHN('./dataset/svhn', download=True, transform=transform_svhn)
    device = 'cuda'
    Fe = F_extractor().to(device)
    F1 = F_label().to(device)
    F2 = F_label().to(device)
    Ft = F_label().to(device)
    batch_size = 128
    # device = 'cuda' if config.mode=='gpu' else 'cpu'
    # print(mnist.__add__(torch.Tensor))
    print(svhn.data.shape)
    # x=torch.utils.data.TensorDataset(mnist.train_data, mnist.train_labels)
    # print(mnist.train_labels)

    train_data = torch.utils.data.DataLoader(mnist,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)


    train_data = torch.utils.data.DataLoader(mnist,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)

    val_data = torch.utils.data.DataLoader(mnist_val,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)

    train_1('cuda', Fe, F1, F2, Ft, train_data, val_data, epochs=10)
    #train_2('cuda', Fe, F1, F2, Ft, None, svhn, None)
    # tmp_1 = []
    # tmp_2 = []
    # tmp_t = []
    # with tqdm(total=len(val_data.dataset), desc='Calculate validation accuracy') as pval:
    #     for idv, (xv, yv) in enumerate(val_data):
    #         xv = xv.to(device)
    #         yv = yv.to(device)
    #         tfeatures = Fe(xv)
    #         tmp_t.append((Ft(tfeatures).max(1)[1] == yv))
    #         tmp_1.append((F1(tfeatures).max(1)[1] == yv))
    #         tmp_2.append((F1(tfeatures).max(1)[1] == yv))
    #         pval.update(val_data.batch_size)
    # val_acc_t = torch.mean(torch.cat(tmp_t).float())
    # val_acc_1 = torch.mean(torch.cat(tmp_1).float())
    # val_acc_2 = torch.mean(torch.cat(tmp_2).float())
    # print(val_acc_t, val_acc_1, val_acc_2)


if __name__ == '__main__':
    main()
