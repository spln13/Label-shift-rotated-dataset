import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

def load_cifar10(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return cifar10_dataset

def load_mnist(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_dataset = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    return mnist_dataset


def create_label_shift_rotated_mnist_dataset(dataset, nums_clients=20, label_per_client=2, train=True):
    rotated_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation([180, 180]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    normal_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Preparation for indices
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, targets = next(iter(data_loader))
    indices_per_class = {i: np.where(targets.numpy() == i)[0] for i in range(10)}
    
    # Divide data
    images_per_label = len(dataset) // 10
    client_per_cluster = nums_clients // (10 // label_per_client)
    client_images_per_label = images_per_label // client_per_cluster
    
    for i in range(nums_clients):
        client_data = []
        start_label = (i // client_per_cluster) * label_per_client
        client_labels = [start_label + k for k in range(label_per_client)]
        start_idx = (i % client_per_cluster) * client_images_per_label
        end_idx = start_idx + client_images_per_label
        
        for label in client_labels:
            label_indices = indices_per_class[label][start_idx:end_idx]
            for idx in label_indices:
                image, target = data[idx], targets[idx]
                if i % 2 == 0:
                    image = rotated_transform(image)
                else:
                    image = normal_transform(image)
                client_data.append((image, target))
        
        # Save data
        client_path = f'./noniid_mnist/client_{i}/'
        os.makedirs(client_path, exist_ok=True)
        if train:
            torch.save(client_data, os.path.join(client_path, 'train_data.pt'))
        else:
            torch.save(client_data, os.path.join(client_path, 'test_data.pt'))




def create_label_shift_rotated_cifar10_dataset(dataset, nums_clients=20, label_per_client=2, train=True):
    rotated_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation([180, 180]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    normal_transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Preparation for indices
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, targets = next(iter(data_loader))
    indices_per_class = {i: np.where(targets.numpy() == i)[0] for i in range(10)}
    
    # Divide data
    images_per_label = len(dataset) // 10
    client_per_cluster = nums_clients // (10 // label_per_client)
    client_images_per_label = images_per_label // client_per_cluster
    
    for i in range(nums_clients):
        client_data = []
        start_label = (i // client_per_cluster) * label_per_client
        client_labels = [start_label + k for k in range(label_per_client)]
        start_idx = (i % client_per_cluster) * client_images_per_label
        end_idx = start_idx + client_images_per_label
        
        for label in client_labels:
            label_indices = indices_per_class[label][start_idx:end_idx]
            for idx in label_indices:
                image, target = data[idx], targets[idx]
                if i % 2 == 0:
                    image = rotated_transform(image)
                else:
                    image = normal_transform(image)
                client_data.append((image, target))
        
        # Save data
        client_path = f'./noniid_cifar10/client_{i}/'
        os.makedirs(client_path, exist_ok=True)
        if train:
            torch.save(client_data, os.path.join(client_path, 'train_data.pt'))
        else:
            torch.save(client_data, os.path.join(client_path, 'test_data.pt'))

def main():
    cifar10_train = load_cifar10(train=True)
    cifar10_test = load_cifar10(train=False)
    create_label_shift_rotated_cifar10_dataset(cifar10_train, nums_clients=20, label_per_client=2, train=True)
    create_label_shift_rotated_cifar10_dataset(cifar10_test, nums_clients=20, label_per_client=2, train=False)

    mnist_train = load_mnist(train=True)
    mnist_test = load_mnist(train=False)
    create_label_shift_rotated_mnist_dataset(mnist_train, nums_clients=20, label_per_client=2, train=True)
    create_label_shift_rotated_mnist_dataset(mnist_test, nums_clients=20, label_per_client=2, train=False)



if __name__ == "__main__":
    main()
