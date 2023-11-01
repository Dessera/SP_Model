from typing import Tuple
import torch
import numpy as np
from torch.utils import data

from config import DATA_PATH, TEST_DATA_PERCENT


class DiabetesDataset(data.Dataset):
    m_x: torch.Tensor
    m_y: torch.Tensor

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__()
        self.m_x = x
        self.m_y = y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.m_x[index], self.m_y[index]

    def __len__(self) -> int:
        return len(self.m_x)


def get_diabetes_data(
    device: torch.device | str,
) -> Tuple[DiabetesDataset, DiabetesDataset]:
    # load data
    data_path = DATA_PATH
    data_np = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
    # split data
    x_np = data_np[:, 0:-1]
    y_np = data_np[:, [-1]]
    # convert to tensor
    # x = torch.from_numpy(x_np)
    # y = torch.from_numpy(y_np)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    # split train/test
    train_size = int(len(x) * (1 - TEST_DATA_PERCENT))
    test_size = len(x) - train_size
    train_x, test_x = torch.split(x, [train_size, test_size])
    train_y, test_y = torch.split(y, [train_size, test_size])
    # create dataset
    train_dataset = DiabetesDataset(train_x, train_y)
    test_dataset = DiabetesDataset(test_x, test_y)
    return train_dataset, test_dataset


# test (available only pwd is project root)
if __name__ == "__main__":
    train_dataset, test_dataset = get_diabetes_data("cpu")
    print(train_dataset[0])
    print(test_dataset[0])
    print(len(train_dataset))
    print(len(test_dataset))
