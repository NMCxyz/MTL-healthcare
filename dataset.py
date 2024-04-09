import torch

from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultitaskDataset(Dataset):
    def __init__(self, features, cls_target, reg_target):
        super().__init__()
        self.x = features
        self.y_cls = cls_target
        self.y_reg = reg_target

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        feature = self.x[idx]
        cls_target = self.y_cls[idx]
        reg_target = self.y_reg[idx]
        return feature.to(device), cls_target.to(device), reg_target.to(device)


class ClassifyDataset(Dataset):
    def __init__(self, features, cls_target):
        super().__init__()
        self.x = features
        self.y_cls = cls_target

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        feature = self.x[idx]
        cls_target = self.y_cls[idx]
        return feature.to(device), cls_target.to(device)
    

class RegressionDataset(Dataset):
    def __init__(self, features, reg_target):
        super().__init__()
        self.x = features
        self.y_reg = reg_target

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        feature = self.x[idx]
        reg_target = self.y_reg[idx]
        return feature.to(device), reg_target.to(device)



def get_data(x_train, y_train, x_test, y_test):
    return x_train, y_train, x_test, y_test

def get_data_mtl(data):
    data = dict(data)
    x_train_npy, y_train_npy, x_test_npy, y_test_npy = get_data(**data)
    tensor_data = {
        "x_train": torch.tensor(x_train_npy[:, : , :-1]).float(),
        "y_train_cls": torch.tensor(x_train_npy[:, : , -1], dtype=torch.long),
        "y_train_reg": torch.tensor(y_train_npy).float(),
        "x_test": torch.tensor(x_test_npy[:, :, :-1]).float(),
        "y_test_cls": torch.tensor(x_test_npy[:, : , -1], dtype=torch.long),
        "y_test_reg": torch.tensor(y_test_npy).float()
    }
    return tensor_data