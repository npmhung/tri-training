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

DATA_WORKERS=2


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
            val_acc_t = val_acc_1 = val_acc_2 = 0
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


def label_target(Fe, F1, F2, T_loader, S_loader, Nt=5000, thres=0.9, device='cuda'):
    T_new = []
    with tqdm(total=len(T_loader.dataset), desc='Labeling target data.') as pbar:
        for idx, (x, _) in enumerate(T_loader):
            """
                assume that we don't have the target dataset's label 
            """
            # print(len(x))
            x = x.to(device)

            y_1 = F1(Fe(x)).max(1)
            y_2 = F2(Fe(x)).max(1)

            cond = (y_1[1] == y_2[1])
            max_prob = torch.stack([y_1[0], y_2[0]], 1).max(1)[0] * cond.float()

            _tmp = [(newx, newy) for (newx, newy, prob) in zip(x, y_1[1], max_prob) if prob > thres]
            T_new.extend(_tmp)
            pbar.update(T_loader.batch_size)

    T_new = T_new[:Nt]
    try:
        T_new_loader = torch.utils.data.TensorDataset(zip(*T_new))
    except:
        T_new_loader = torch.utils.data.TensorDataset([], [])

    L = Concat2Dataset(S_loader, T_new_loader)
    L_loader = torch.utils.data.DataLoader(L, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)

    return L_loader, T_new_loader


def train_2(device, Fe, F1, F2, Ft, S, T, Val, step=50, iters = 2, val_step=100):
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

    # label
    T_loader = torch.utils.data.DataLoader(T, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)
    S_loader = torch.utils.data.DataLoader(S, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)
    Val_loader = torch.utils.data.DataLoader(Val, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)
    L_loader, T_new_loader = label_target(Fe, F1, F2, T_loader, S_loader, 5000, thres=0.8)

    thres = 0.81
    opt_label = optim.Adam([*Fe.parameters(), *F1.parameters(), *F2.parameters()])
    scheduler_1 = optim.lr_scheduler.StepLR(opt_label, 100)
    opt_target = optim.Adam([*Fe.parameters(), *Ft.parameters()])
    scheduler_2 = optim.lr_scheduler.StepLR(opt_target, 100)
    W1 = next(F1.parameters())
    W2 = next(F2.parameters())
    lambda_view = 1e-2

    for k in range(step):
        print('New target dataset size %d'%(len(T_new_loader.dataset)))
        thres = min(0.9, thres+0.01)
        for _ in range(iters):
            with tqdm(total=len(L_loader.dataset), desc='Train labeling network.') as pbar:
                train_acc1 = train_acc2 = val_acc1 = val_acc2 = 0
                for idx, (x, y) in enumerate(L_loader):
                    scheduler_1.step()
                    x = x.to(device)
                    y = y.to(device)

                    features = Fe(x)
                    o1 = F1(features)
                    o2 = F2(features)

                    view_loss = lambda_view * torch.sum(torch.abs(W1 * W2))
                    loss = F.cross_entropy(o1, y) + F.cross_entropy(o2, y) + view_loss

                    opt_label.zero_grad()
                    loss.backward()
                    opt_label.step()

                    if idx % val_step ==0:
                        tmp1 = []
                        tmp2 = []
                        for idy, (xval, yval) in enumerate(Val_loader):
                            xval = xval.to(device)
                            yval = yval.to(device)

                            _features = Fe(xval)
                            _o1 = F1(_features)
                            _o2 = F2(_features)

                            tmp1.append(_o1.max(1)[1] == yval)
                            tmp2.append(_o2.max(1)[1] == yval)

                        val_acc1 = torch.mean(torch.cat(tmp1).float())
                        val_acc2 = torch.mean(torch.cat(tmp2).float())

                    train_acc1 = torch.mean((o1.max(1)[1] == y).float())
                    train_acc2 = torch.mean((o2.max(1)[1] == y).float())

                    pbar.set_postfix_str('Train acc 1: %f - 2: %f \n Validation acc 1: %f - 2: %f' %
                                         (train_acc1, train_acc2, val_acc1, val_acc2))
                    pbar.update(L_loader.batch_size)

            with tqdm(total=len(T_new_loader.dataset), desc='Train target network.') as pbar:
                train_acc = val_acc = 0
                for idx, (x, y) in enumerate(T_new_loader):
                    x = x.to(device)
                    y = y.to(device)

                    o = Ft(Fe(x))

                    loss = F.cross_entropy(o, y)

                    opt_target.zero_grad()
                    loss.backward()
                    opt_target.step()

                    if idx % val_step:
                        tmp = []
                        for idy, (xval, yval) in enumerate(Val_loader):
                            xval = xval.to(device)
                            yval = yval.to(device)

                            _o = Ft(Fe(xval))
                            tmp.append(_o.max(1)[1] == yval)

                        val_acc = torch.mean(torch.cat(tmp).float())

                    pbar.set_postfix_str('Train acc: %f - Validation acc: %f' % (train_acc, val_acc))

        Nt = int(k/20*len(T_loader))
        L_loader, T_new_loader = label_target(Fe, F1, F2, T_loader, S_loader, Nt, thres=thres)
        save_checkpoint({
            'epoch': k + 1,
            'Fe': Fe.state_dict(),
            'F1': F1.state_dict(),
            'F2': F2.state_dict(),
            'Ft': Ft.state_dict(),
            'opt_label': opt_label.state_dict(),
            'opt_target': opt_target.state_dict(),
            'scheduler_1': scheduler_1.state_dict(),
            'scheduler_2': scheduler_2.state_dict()
        }, False, 'phase2.pth.tar')


def main():
    transform_mnist = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    transform_svhn = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    mnist = tv.datasets.MNIST('./dataset/mnist', download=True, transform=transform_mnist, train=True)
    mnist_val = tv.datasets.MNIST('./dataset/mnist', download=True, transform=transform_mnist, train=False)
    svhn = tv.datasets.SVHN('./dataset/svhn', download=True, transform=transform_svhn, split='train')
    svhn_val = tv.datasets.SVHN('./dataset/svhn', download=True, transform=transform_svhn, split='test')
    device = 'cuda'
    models = get_model()
    Fe = models['Fe']
    F1 = models['F1']
    F2 = models['F2']
    Ft = models['Ft']
    batch_size = 128
    # device = 'cuda' if config.mode=='gpu' else 'cpu'
    # print(mnist.__add__(torch.Tensor))
    print(svhn.data.shape)
    # x=torch.utils.data.TensorDataset(mnist.train_data, mnist.train_labels)
    # print(mnist.train_labels)

    train_data = torch.utils.data.DataLoader(mnist,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=DATA_WORKERS)

    val_data = torch.utils.data.DataLoader(mnist_val,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=DATA_WORKERS)

    train_1('cuda', Fe, F1, F2, Ft, train_data, val_data, epochs=1)
    train_2('cuda', Fe, F1, F2, Ft, mnist, svhn, svhn_val)


if __name__ == '__main__':
    main()
