import torchvision.transforms.transforms as transforms
import torch
import cv2 as cv
import torchvision.transforms as transforms
from torch.utils.data import Dataset as pytorch_dataset
import glob
from PIL import Image
import pandas as pd
from config import *
import re
import numpy as np
blank_img=np.zeros([3,128,128])

def keyword(path):
    return int(path.split(os.path.sep)[-1].split('.')[0])

def img_order(path):
    return int(path.split(os.path.sep)[-1].split('.')[0].split("_")[-2])

def t2_order(path):
    return int(path.split(os.path.sep)[-1].split('.')[0].split("_")[-1])

excel_dir=r"label.xlsx"
time_excel=r"time_label.xlsx"
time_content=pd.read_excel(time_excel)
class MyDataset(pytorch_dataset):
    def __init__(self, path,sheet_name,transform=None):
        super(MyDataset, self).__init__()
        work_dir = path
        all_path = []
        self.datadir = work_dir
        for dir in work_dir:
            id_path = glob.glob(os.path.join(dir, "*"))
            all_path.extend(id_path)
        self.data_path = all_path
        self.transform = transform
        self.excel_content=pd.read_excel(excel_dir,sheet_name=sheet_name)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        P_img = []
        T2_img = []
        DWI_img = []
        CEUS_img = []
        path_2D = self.data_path[item]
        ipath = path_2D

        part = ipath.split(os.path.sep)[-1]
        print(part)
        group = ipath.split(os.path.sep)[-3]

        for item in range(len(time_content['ID'])):
            if re.search(str(r'.*' + str(time_content['ID'][item])), part):
                label = time_content['Label'][item]
        for item in range(len(time_content['ID'])):
            if re.search(str(r'.*' + str(time_content['ID'][item])), part):
                time =time_content['Time'][item]
        unified_2D_p = transforms.Compose([
            transforms.Resize(p_shape),
            transforms.ToTensor(),
        ]
        )
        unified_2D_mr = transforms.Compose([
            transforms.Resize(mr_shape),
            transforms.ToTensor(),
        ]
        )

        modality_list = glob.glob(os.path.join(ipath, "*"))
        modality = []
        for l in modality_list:
            modality.append(l.split(os.path.sep)[-1])
        # if "P" in modality:
        p_path = glob.glob(os.path.join(ipath, "P", "*_1.jpg"))
        p_path.sort(key=img_order)
        if len(p_path) >= 2:
            for num in range(0, 2):
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_p(us_img)
                P_img.append(us_img)
        elif len(p_path) == 1:
            us_img = cv.imread(p_path[0])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            P_img.append(us_img)
            us_img = cv.imread(p_path[0])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            P_img.append(us_img)
        else:
            P_img.append(blank_img)
            P_img.append(blank_img)

        p_path = glob.glob(os.path.join(ipath, "T2", "*"))
        p_path.sort(key=t2_order)
        if len(p_path) >= 3:
            margin = len(p_path) // 3
            num=0
            i=0
            while num <len(p_path) and i<3:
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                T2_img.append(us_img)
                num+=margin
                i+=1
        elif len(p_path) == 2:
            for num in range(2):
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                T2_img.append(us_img)
            us_img = cv.imread(p_path[1])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_mr(us_img)
            T2_img.append(us_img)
        elif len(p_path) == 1:
            for num in range(3):
                us_img = cv.imread(p_path[0])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                T2_img.append(us_img)
        else:
            T2_img.append(blank_img)
            T2_img.append(blank_img)
            T2_img.append(blank_img)
        # if "DWI" in modality:
        #     DWI_img = []
        p_path = glob.glob(os.path.join(ipath, "DWI", "*"))
        p_path.sort(key=t2_order)
        if len(p_path) >= 3:
            margin = len(p_path) // 3
            num = 0
            i = 0
            while num < len(p_path) and i<3:
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                DWI_img.append(us_img)
                num+=margin
                i+=1
        elif len(p_path) == 2:
            for num in range(2):
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                DWI_img.append(us_img)
            us_img = cv.imread(p_path[1])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_mr(us_img)
            DWI_img.append(us_img)
        elif len(p_path) == 1:
            for num in range(3):
                us_img = cv.imread(p_path[0])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                DWI_img.append(us_img)
        else:
            DWI_img.append(blank_img)
            DWI_img.append(blank_img)
            DWI_img.append(blank_img)

        p_path = glob.glob(os.path.join(ipath, "CEUS", "*.jpg"))
        p_path.sort(key=img_order)
        if len(p_path) >= 3:
            for num in range(0, 3):
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_p(us_img)
                CEUS_img.append(us_img)
        elif len(p_path) == 2:
            us_img = cv.imread(p_path[0])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            CEUS_img.append(us_img)
            us_img = cv.imread(p_path[1])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            CEUS_img.append(us_img)
            us_img = cv.imread(p_path[1])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            CEUS_img.append(us_img)
        elif len(p_path) == 1:
            for j in range(3):
                us_img = cv.imread(p_path[0])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_p(us_img)
                CEUS_img.append(us_img)

        else:
            CEUS_img.append(blank_img)
            CEUS_img.append(blank_img)
            CEUS_img.append(blank_img)

        P_img = np.stack(P_img, axis=0)
        P_img = np.array(P_img, dtype=np.float32)
        P_img = torch.from_numpy(P_img)
        P_img = torch.transpose(P_img, 0, 1)

        T2_img = np.stack(T2_img, axis=0)
        T2_img = np.array(T2_img, dtype=np.float32)
        T2_img = torch.from_numpy(T2_img)
        T2_img = torch.transpose(T2_img, 0, 1)

        DWI_img = np.stack(DWI_img, axis=0)
        DWI_img = np.array(DWI_img, dtype=np.float32)
        DWI_img = torch.from_numpy(DWI_img)
        DWI_img = torch.transpose(DWI_img, 0, 1)

        CEUS_img = np.stack(CEUS_img, axis=0)
        CEUS_img = np.array(CEUS_img, dtype=np.float32)
        CEUS_img = torch.from_numpy(CEUS_img)
        CEUS_img = torch.transpose(CEUS_img, 0, 1)

        return part, group, label, P_img, T2_img, DWI_img, CEUS_img,time

class MyDataset_extra_SEC(pytorch_dataset):
    def __init__(self, path, sheet_name, transform=None):
        super(MyDataset_extra_SEC, self).__init__()
        work_dir = path

        all_path = []
        self.datadir = work_dir
        for dir in work_dir:
            id_path = glob.glob(os.path.join(dir, "*"))
            all_path.extend(id_path)
        self.data_path = all_path
        self.transform = transform
        self.excel_content = pd.read_excel(excel_dir, sheet_name=sheet_name)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        P_img = []
        T2_img = []
        DWI_img = []
        CEUS_img = []
        path_2D = self.data_path[item]
        ipath = path_2D
        part = ipath.split(os.path.sep)[-1].split("_")[0]
        print(part)
        group = ipath.split(os.path.sep)[-3]
        for item in range(len(time_content['ID'])):
            if re.search(str(r'.*' + str(time_content['ID'][item])), part):
                label = time_content['Label'][item]
        for item in range(len(time_content['ID'])):
            if re.search(str(r'.*' + str(time_content['ID'][item])), part):
                time = time_content['Time'][item]
        unified_2D_p = transforms.Compose([
            transforms.Resize(p_shape),
            transforms.ToTensor(),
        ]
        )
        unified_2D_mr = transforms.Compose([
            transforms.Resize(mr_shape),
            transforms.ToTensor(),
        ]
        )

        modality_list = glob.glob(os.path.join(ipath, "*"))
        modality = []
        for l in modality_list:
            modality.append(l.split(os.path.sep)[-1])
        p_path=[]
        path = glob.glob(os.path.join(ipath, "P", "*.jpg"))
        for p in path:
            if "SEC" in p:
                p_path.append(p)
        p_path.sort(key=img_order)
        if len(p_path) >= 2:
            for num in range(0, 2):
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_p(us_img)
                P_img.append(us_img)
        elif len(p_path) == 1:
            us_img = cv.imread(p_path[0])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            P_img.append(us_img)
            us_img = cv.imread(p_path[0])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            P_img.append(us_img)
        else:
            P_img.append(blank_img)
            P_img.append(blank_img)


        p_path = []
        path = glob.glob(os.path.join(ipath, "T2", "*"))
        for p in path:
            if "SEC" in p:
                p_path.append(p)
        p_path.sort(key=t2_order)
        if len(p_path) >= 3:
            margin = len(p_path) // 3
            num = 0
            i = 0
            while num < len(p_path) and i < 3:
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                T2_img.append(us_img)
                num += margin
                i += 1
        elif len(p_path) == 2:
            for num in range(2):
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                T2_img.append(us_img)
            us_img = cv.imread(p_path[1])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_mr(us_img)
            T2_img.append(us_img)
        elif len(p_path) == 1:
            for num in range(3):
                us_img = cv.imread(p_path[0])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                T2_img.append(us_img)
        else:
            T2_img.append(blank_img)
            T2_img.append(blank_img)
            T2_img.append(blank_img)
        p_path = []
        path = glob.glob(os.path.join(ipath, "DWI", "*"))
        for p in path:
            if "SEC" in p:
                p_path.append(p)
        p_path.sort(key=t2_order)
        if len(p_path) >= 3:
            margin = len(p_path) // 3
            num = 0
            i = 0
            while num < len(p_path) and i < 3:
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                DWI_img.append(us_img)
                num += margin
                i += 1
        elif len(p_path) == 2:
            for num in range(2):
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                DWI_img.append(us_img)
            us_img = cv.imread(p_path[1])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_mr(us_img)
            DWI_img.append(us_img)
        elif len(p_path) == 1:
            for num in range(3):
                us_img = cv.imread(p_path[0])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_mr(us_img)
                DWI_img.append(us_img)
        else:
            DWI_img.append(blank_img)
            DWI_img.append(blank_img)
            DWI_img.append(blank_img)

        p_path = []
        path = glob.glob(os.path.join(ipath, "CEUS", "*"))
        for p in path:
            if "SEC" in p:
                p_path.append(p)
        p_path.sort(key=img_order)
        if len(p_path) >= 3:
            for num in range(0, 3):
                us_img = cv.imread(p_path[num])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_p(us_img)
                CEUS_img.append(us_img)
        elif len(p_path) == 2:
            us_img = cv.imread(p_path[0])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            CEUS_img.append(us_img)
            us_img = cv.imread(p_path[1])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            CEUS_img.append(us_img)
            us_img = cv.imread(p_path[1])
            us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
            us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
            us_img = Image.fromarray(us_img.astype('uint8'))
            us_img = unified_2D_p(us_img)
            CEUS_img.append(us_img)
        elif len(p_path) == 1:
            for j in range(3):
                us_img = cv.imread(p_path[0])
                us_img = cv.cvtColor(us_img, cv.COLOR_RGB2GRAY)
                us_img = cv.cvtColor(us_img, cv.COLOR_GRAY2BGR)
                us_img = Image.fromarray(us_img.astype('uint8'))
                us_img = unified_2D_p(us_img)
                CEUS_img.append(us_img)

        else:
            CEUS_img.append(blank_img)
            CEUS_img.append(blank_img)
            CEUS_img.append(blank_img)

        P_img = np.stack(P_img, axis=0)
        P_img = np.array(P_img, dtype=np.float32)
        P_img = torch.from_numpy(P_img)
        P_img = torch.transpose(P_img, 0, 1)

        T2_img = np.stack(T2_img, axis=0)
        T2_img = np.array(T2_img, dtype=np.float32)
        T2_img = torch.from_numpy(T2_img)
        T2_img = torch.transpose(T2_img, 0, 1)

        DWI_img = np.stack(DWI_img, axis=0)
        DWI_img = np.array(DWI_img, dtype=np.float32)
        DWI_img = torch.from_numpy(DWI_img)
        DWI_img = torch.transpose(DWI_img, 0, 1)

        CEUS_img = np.stack(CEUS_img, axis=0)
        CEUS_img = np.array(CEUS_img, dtype=np.float32)
        CEUS_img = torch.from_numpy(CEUS_img)
        CEUS_img = torch.transpose(CEUS_img, 0, 1)

        return part, group, label, P_img, T2_img, DWI_img, CEUS_img,time
if __name__ == '__main__':
    dataset = MyDataset(path=["Train"])
    for part, group, label, P_img, T2_img, DWI_img, CEUS_img,time in dataset:
        print(part,group,label,P_img.shape,T2_img.shape,DWI_img.shape,CEUS_img.shape,time)
