import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Tuple, Dict, Optional
import os


class FEMNISTDataset(Dataset):
    """FEMNIST dataset for federated learning experiments."""

    def __init__(self, data_dir: str, train: bool = True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        # Note: This is a placeholder - actual FEMNIST loading would require
        # downloading and preprocessing the LEAF dataset
        self.data = []
        self.targets = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target


def load_cifar10(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    return trainset, testset


def load_cifar100(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load CIFAR-100 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    return trainset, testset


def create_iid_splits(dataset: Dataset, num_clients: int, seed: int = 42) -> List[Subset]:
    """Create IID data splits for clients."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    client_size = len(dataset) // num_clients

    client_datasets = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else len(dataset)
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))

    return client_datasets


def create_dirichlet_splits(dataset: Dataset, num_clients: int, alpha: float = 0.5,
                          num_classes: int = 10, seed: int = 42) -> List[Subset]:
    """
    Create non-IID data splits using Dirichlet distribution.

    Args:
        dataset: The dataset to split
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        num_classes: Number of classes in the dataset
        seed: Random seed
    """
    np.random.seed(seed)

    # Get labels from dataset
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Extract labels by iterating through dataset
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Create class-wise indices
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Sample proportions for each client using Dirichlet distribution
    client_datasets = []
    for client_id in range(num_clients):
        client_indices = []
        proportions = np.random.dirichlet([alpha] * num_classes)

        for class_id in range(num_classes):
            num_samples = int(proportions[class_id] * len(class_indices[class_id]) / num_clients)
            if num_samples > 0:
                sampled_indices = np.random.choice(
                    class_indices[class_id], size=num_samples, replace=False
                )
                client_indices.extend(sampled_indices)
                # Remove sampled indices to avoid overlap
                class_indices[class_id] = np.setdiff1d(class_indices[class_id], sampled_indices)

        if len(client_indices) > 0:
            client_datasets.append(Subset(dataset, client_indices))
        else:
            # Fallback: give at least one sample per client
            remaining_indices = np.concatenate(class_indices)
            if len(remaining_indices) > 0:
                sample_idx = np.random.choice(remaining_indices, 1)
                client_datasets.append(Subset(dataset, sample_idx))
            else:
                client_datasets.append(Subset(dataset, [0]))  # Emergency fallback

    return client_datasets


def get_class_distribution(dataset: Subset) -> Dict[int, int]:
    """Get class distribution for a dataset subset."""
    if hasattr(dataset.dataset, 'targets'):
        all_targets = np.array(dataset.dataset.targets)
    else:
        all_targets = np.array([dataset.dataset[i][1] for i in range(len(dataset.dataset))])

    subset_targets = all_targets[dataset.indices]
    unique, counts = np.unique(subset_targets, return_counts=True)
    return dict(zip(unique, counts))


def analyze_data_distribution(client_datasets: List[Subset], num_classes: int = 10) -> Dict:
    """Analyze the data distribution across clients."""
    analysis = {
        'total_samples': sum(len(ds) for ds in client_datasets),
        'samples_per_client': [len(ds) for ds in client_datasets],
        'class_distributions': [get_class_distribution(ds) for ds in client_datasets],
    }

    # Calculate statistics
    samples_per_client = analysis['samples_per_client']
    analysis['mean_samples_per_client'] = np.mean(samples_per_client)
    analysis['std_samples_per_client'] = np.std(samples_per_client)

    # Calculate class distribution statistics
    class_counts = np.zeros((len(client_datasets), num_classes))
    for i, dist in enumerate(analysis['class_distributions']):
        for class_id, count in dist.items():
            class_counts[i, class_id] = count

    analysis['class_counts_matrix'] = class_counts
    analysis['classes_per_client'] = [(class_counts[i] > 0).sum() for i in range(len(client_datasets))]
    analysis['mean_classes_per_client'] = np.mean(analysis['classes_per_client'])

    return analysis


def create_dataloaders(client_datasets: List[Subset], batch_size: int = 32,
                      shuffle: bool = True) -> List[DataLoader]:
    """Create DataLoaders for client datasets."""
    return [DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
            for ds in client_datasets]