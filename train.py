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

DATA_WORKERS=8


def train_1(device, models, train_loader, val_loader, epochs=1, restore=False, path=None):
    opt = optim.Adam([*models['F1'].parameters(), *models['F2'].parameters(), *models['Fe'].parameters(), *models['Ft'].parameters()], lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, 100, 0.9)
    
    models['optimizer'] = opt
    models['scheduler'] = scheduler

    if restore:
        models = get_model(models, pretrain=True, path=path)

    F1 = models['F1']
    F2 = models['F2']
    Fe = models['Fe']
    Ft = models['Ft']
    opt = models['optimizer']
    scheduler = models['scheduler']
    lambda_view = 1e-2
    W1 = next(F1.parameters())
    W2 = next(F2.parameters())
    for epoch in range(epochs):
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
                view_loss = lambda_view * torch.sum(torch.abs(W1.transpose(0,1)@W2))
                loss = F.cross_entropy(f1_out, y) + F.cross_entropy(f2_out, y) + view_loss + F.cross_entropy(ft_out, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                acc = (Ft(Fe(x)).max(1)[1] == y).sum().item() / train_loader.batch_size

                tmp_1 = []
                tmp_2 = []
                tmp_t = []
                if idx % 100 == 0:
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

                pbar.set_postfix_str('Loss %f - View Loss %f - Target net Acc %f - Net 1 Acc %f - Net 2 Acc %f'%(loss, view_loss, val_acc_t, val_acc_1, val_acc_2))
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

    T_new_loader = None
    try:
        xx, yy = [*zip(*T_new)]

        xx = torch.stack(xx, 0).to('cpu')
        yy = torch.stack(yy, 0).to('cpu')
    except Exception as e:
        print(e)
        xx = torch.tensor([])
        yy = torch.tensor([])
    T_new = torch.utils.data.TensorDataset(xx, yy)
    #except Exception as e:
    #    print(e)


    # if T_new_loader is None:
    #     L = S_loader.dataset
    L = Concat2Dataset(S_loader.dataset, T_new) if T_new is not None else S_loader.dataset
    L_loader = torch.utils.data.DataLoader(L, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)
    T_new_loader = torch.utils.data.DataLoader(T_new, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)

    return L_loader, T_new_loader


def train_2(device, models,  S, T, Val, step=50, iters = 2, val_step=100, restore=False, path=None):
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

    thres = 0.95
    opt_label = optim.Adam([*models['Fe'].parameters(), *models['F1'].parameters(), *models['F2'].parameters()], lr=1e-5)
    #scheduler_1 = optim.lr_scheduler.StepLR(opt_label, 100, gamma=0.99)
    opt_target = optim.Adam([*models['Fe'].parameters(), *models['Ft'].parameters()], lr=1e-4)
    #scheduler_2 = optim.lr_scheduler.StepLR(opt_target, 100, gamma=0.99)
    
    models['opt_label'] = opt_label
    models['opt_target'] = opt_target
    models['step'] = 0
    if restore:
        assert path is not None, "Please provide path of pretrain model"
        models = get_model(models, pretrain=True, path=path)

    F1 = models['F1']
    F2 = models['F2']
    Fe = models['Fe']
    Ft = models['Ft']
    opt_label = models['opt_label']
    opt_target = models['opt_target']

    # label
    T_loader = torch.utils.data.DataLoader(T, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)
    S_loader = torch.utils.data.DataLoader(S, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)
    Val_loader = torch.utils.data.DataLoader(Val, batch_size=128, num_workers=DATA_WORKERS, shuffle=True)
    L_loader, T_new_loader = label_target(Fe, F1, F2, T_loader, S_loader, 5000, thres=0.8)

    W1 = next(F1.parameters())
    W2 = next(F2.parameters())
    lambda_view = 1e-2
    print('Model step ', models['step'])
    for k in range(models['step'], step):
        try:
            print('New target dataset size %d'%(len(T_new_loader.dataset)))
        except:
            pass
        thres = min(0.95, thres+0.01)
        for _ in range(iters):
            with tqdm(total=len(L_loader.dataset), desc='Train labeling network.') as pbar:
                train_acc1 = train_acc2 = val_acc1 = val_acc2 = 0
                for idx, (x, y) in enumerate(L_loader):
                    #scheduler_1.step()
                    x = x.to(device)
                    y = y.to(device)

                    features = Fe(x)
                    o1 = F1(features)
                    o2 = F2(features)

                    view_loss = lambda_view * torch.sum(torch.abs(W1.transpose(0, 1) @ W2))
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

                    pbar.set_postfix_str('Loss %f - View Loss %f - Train acc 1: %f - 2: %f Validation acc 1: %f - 2: %f' %
                                         (loss, view_loss, train_acc1, train_acc2, val_acc1, val_acc2))
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

                    if idx % val_step == 0:
                        tmp = []
                        for idy, (xval, yval) in enumerate(Val_loader):
                            xval = xval.to(device)
                            yval = yval.to(device)

                            _o = Ft(Fe(xval))
                            tmp.append(_o.max(1)[1] == yval)

                        val_acc = torch.mean(torch.cat(tmp).float())
                    train_acc = torch.mean((o.max(1)[1] == y).float())
                    pbar.set_postfix_str('Loss %f - Train acc: %f - Validation acc: %f' % (loss, train_acc, val_acc))
                    pbar.update(T_new_loader.batch_size)

        Nt = int((k+1)/20*len(T_loader.dataset))
        print('NT ', Nt, len(T_loader))
        L_loader, T_new_loader = label_target(Fe, F1, F2, T_loader, S_loader, Nt, thres=thres)
        save_checkpoint({
            'step': k + 1,
            'Fe': Fe.state_dict(),
            'F1': F1.state_dict(),
            'F2': F2.state_dict(),
            'Ft': Ft.state_dict(),
            'opt_label': opt_label.state_dict(),
            'opt_target': opt_target.state_dict()
            #'scheduler_1': scheduler_1.state_dict(),
            #'scheduler_2': scheduler_2.state_dict()
        }, False, 'phase2.pth.tar')


def main():
    transform_mnist = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    transform_svhn = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    mnist = tv.datasets.MNIST('./dataset/mnist', download=False, transform=transform_mnist, train=True)
    mnist_val = tv.datasets.MNIST('./dataset/mnist', download=False, transform=transform_mnist, train=False)
    svhn = tv.datasets.SVHN('./dataset/svhn', download=False, transform=transform_svhn, split='train')
    svhn_val = tv.datasets.SVHN('./dataset/svhn', download=False, transform=transform_svhn, split='test')
    device = 'cuda'
    Fe = F_extractor().to(device)
    F1 = F_label().to(device)
    F2 = F_label().to(device)
    Ft = F_label().to(device)

    models = {
        'Fe': Fe,
        'F1': F1,
        'F2': F2,
        'Ft': Ft
    }
    # print(mnist.__add__(torch.Tensor))
    print(svhn.data.shape)
    # x=torch.utils.data.TensorDataset(mnist.train_data, mnist.train_labels)
    # print(mnist.train_labels)
    batch_size = 128
    train_data = torch.utils.data.DataLoader(mnist,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=DATA_WORKERS)

    val_data = torch.utils.data.DataLoader(mnist_val,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=DATA_WORKERS)
    train_1('cuda', models, train_data, val_data, epochs=1, restore=False, path='phase1.pth.tar')
    train_2('cuda', models,  mnist, svhn, svhn_val, step=200, val_step=400, iters=2, restore=False, path='phase2.pth.tar')


if __name__ == '__main__':
    main()
