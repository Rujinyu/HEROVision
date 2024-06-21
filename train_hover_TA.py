from Model_hovertrans import FourModelHover
from dataset import MyDataset_extra_SEC,MyDataset
import torch
import sklearn.metrics as metrics
from config import *
import numpy as np
from torch.utils.data.dataloader import DataLoader
from Model_hovertrans import cox_loss



def train_one_epoch(model, optimizer, dataloader, lr_scheduler):
    model.train()
    for part, group, label, p_img, T2_img, DWI_img, ceus_img  in dataloader:
        # B * 1 * C * W * H
        T2_img = T2_img.to(device=device)
        DWI_img = DWI_img.to(device=device)
        p_img = p_img.to(device=device)
        ceus_img = ceus_img.to(device=device)
        label = label.to(device=device, dtype=torch.long)
        y = model(T2_img, DWI_img, p_img, ceus_img)
        l = cox_loss(y, label)
        l.requires_grad_(True)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step(l)

def validate(model, dataloader, epoch, filename):
    model.eval()
    y_true = []
    y_score = []
    l_sum = 0
    sum = 0
    for part, group, label, p_img, T2_img, DWI_img, ceus_img  in dataloader:
        # B * 2 * C * W * H
        T2_img = T2_img.to(device=device)
        DWI_img = DWI_img.to(device=device)
        p_img = p_img.to(device=device)
        ceus_img = ceus_img.to(device=device)
        label = label.to(device=device, dtype=torch.long)
        with torch.no_grad():
            y = model(T2_img,DWI_img,p_img,ceus_img)
        l = cox_loss(y, label)
        sum = sum + 1
        l_sum = l_sum + l.item()
        y_score.append(torch.softmax(y, dim=-1).detach().cpu().numpy())
        y_true.append(label.reshape(-1).detach().cpu().numpy())
    y_true = np.concatenate(y_true, 0)
    y_score = np.concatenate(y_score, 0)
    auc = metrics.roc_auc_score(y_true, y_score[:, 1], multi_class='ovr')
    acc = np.sum(np.argmax(y_score, axis=1) == y_true) / y_true.shape[0]
    print("epoch: {} , loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
        epoch, l_sum / sum, auc, acc))
    with open(filename, 'a') as fp:
        fp.write("epoch: {} , loss: {:.5f}, auc: {:.3f}, acc: {:.3f}".format(
            epoch, l_sum / sum, auc, acc))
        fp.write("\n")


def train(load_state=False):
    dataset_train = MyDataset(path=["Train"])
    dataset_test = MyDataset(path=["Test"])
    dataset_VALI1 = MyDataset(path=["VALI1"])
    dataset_VALI2 = MyDataset(path=["VALI2"])
    dataset_VALI3 = MyDataset_extra_SEC(path=["VALI3"])

    dataloader_train= DataLoader(
        dataset_train, batch_size=16, shuffle=False, pin_memory=False if device == device else True,
        drop_last=False)
    dataloader_test= DataLoader(
        dataset_test, batch_size=8, shuffle=False, pin_memory=False if device == device else True,
        drop_last=False)
    dataloader_VALI1 = DataLoader(
        dataset_VALI1, batch_size=8, shuffle=False, pin_memory=False if device == device else True,
        drop_last=False)
    dataloader_VALI2 = DataLoader(
        dataset_VALI2, batch_size=8, shuffle=False, pin_memory=False if device == device else True,
        drop_last=False)
    dataloader_VALI3 = DataLoader(
        dataset_VALI3, batch_size=8, shuffle=False, pin_memory=False if device == device else True,
        drop_last=False)
    filename = "record.txt"
    model = FourModelHover(img_size=128, patch_size=[16, 8]).to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    torch.manual_seed(42)
    if load_state == True:
        state = torch.load("base.pth")
        model.load_state_dict(state, strict=False)
        print("load finish!")


    for epoch in range(100):
        train_one_epoch(model, optimizer, dataloader_train, lr_scheduler)
        validate(model, dataloader_train, epoch, filename)
        validate(model, dataloader_test, epoch, filename)
        validate(model, dataloader_VALI1, epoch, filename)
        validate(model, dataloader_VALI2, epoch, filename)
        validate(model, dataloader_VALI3, epoch, filename)
        state_dict = model.state_dict()
        torch.save(
            state_dict, "{}.pth".format(epoch))



if __name__ == '__main__':
    train()
