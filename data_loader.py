import torch
from torch.utils.data import Dataset, DataLoader

class CustomizedDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label


def load_data_for_client(client_id, dataset='cifar10'):
    if dataset == 'cifar10':
        data_path = f'./noniid_cifar10/client_{client_id}/train_data.pt'
    else:  # mnist
        data_path = f'./noniid_mnist/client_{client_id}/train_data.pt'
    dataset = CustomizedDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return data_loader

if __name__ == '__main__':
    client_data_loader = load_data_for_client(0, dataset='cifar10')
    # 使用 DataLoader
    for images, labels in client_data_loader:
        print(images.shape, labels.shape)
        break  # 打印第一批数据的维度并停止
