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


class NoisyLabelSubset(Dataset):
    """
    Wrap a client's dataset so a fixed fraction of its labels are flipped to
    a wrong class. Used to simulate an honest-but-noisy client for the pilot
    that asks whether TAVS's trust signal can identify low-quality clients
    at all.

    Contract:
      - noise_fraction of THIS client's samples get a corrupted label; the
        rest are returned untouched. Corrupting per-client, not per-sample,
        so a client's noise profile stays fixed across epochs and rounds --
        that is what BVD would have to spot.
      - Which samples are corrupted is drawn deterministically from `seed`,
        so re-running the same experiment produces the same noisy set. A
        run without this determinism would move the label noise across
        seeds and confound the "does trust identify the noisy client"
        question with dataset-shuffle luck.
      - noise_type controls how the wrong label is chosen:
          "uniform"  : any of the (num_classes - 1) wrong labels, uniform draw.
                       This is the classical symmetric-noise regime -- easy
                       for FedAvg to dampen because per-round wrong-gradients
                       point in random directions and largely cancel.
          "pairflip" : each true class maps to a fixed confusable partner
                       (see pair_map below). Wrong-gradients under pair-flip
                       point in a consistent direction, so they do NOT cancel
                       on aggregation -- this is the regime where a trust
                       signal has something to actually latch onto.
      - The wrapper never mutates the underlying dataset. Two clients that
        share a base dataset are independent: their noise decisions do not
        collide.
    """

    # CIFAR-10 pair-flip map. Standard confusable pairs from the
    # noisy-labels literature (Han et al. NeurIPS'18, Yao et al. NeurIPS'20).
    # Symmetric: 0<->2, 1<->9, 3<->5, 4<->7, 6<->8. Meaning:
    #   airplane <-> bird           (both fly, sky background)
    #   automobile <-> truck        (wheeled vehicles)
    #   cat <-> dog                 (four-legged pets)
    #   deer <-> horse              (four-legged large mammals)
    #   frog <-> ship               (weakest pair; kept for symmetry)
    # Undefined for other class counts -- callers passing num_classes != 10
    # with noise_type="pairflip" must supply their own pair_map.
    _CIFAR10_PAIRFLIP = {0: 2, 2: 0, 1: 9, 9: 1, 3: 5, 5: 3, 4: 7, 7: 4, 6: 8, 8: 6}

    def __init__(self, base: Dataset, noise_fraction: float,
                 num_classes: int = 10, seed: int = 0,
                 noise_type: str = "uniform",
                 pair_map: dict = None):
        if not (0.0 <= noise_fraction <= 1.0):
            raise ValueError(f"noise_fraction must be in [0, 1]; got {noise_fraction}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2; got {num_classes}")
        if noise_type not in ("uniform", "pairflip"):
            raise ValueError(
                f"noise_type must be 'uniform' or 'pairflip'; got {noise_type!r}"
            )

        self.base = base
        self.num_classes = num_classes
        self.noise_type = noise_type

        # Pair-flip map. Default to the CIFAR-10 pattern; callers with a
        # different class count or a different pairing must pass one in.
        if noise_type == "pairflip":
            resolved = pair_map if pair_map is not None else self._CIFAR10_PAIRFLIP
            # Sanity-check the map: keys and values must be valid class
            # ids, and no class may map to itself (that would silently
            # revert some flips to no-ops).
            for src, dst in resolved.items():
                if not (0 <= src < num_classes) or not (0 <= dst < num_classes):
                    raise ValueError(
                        f"pair_map entry {src}->{dst} out of range for "
                        f"num_classes={num_classes}"
                    )
                if src == dst:
                    raise ValueError(
                        f"pair_map has self-loop {src}->{src}; "
                        "self-loops silently reduce effective noise rate"
                    )
            self.pair_map = dict(resolved)
        else:
            self.pair_map = None

        n = len(base)
        n_noisy = int(round(noise_fraction * n))

        rng = np.random.default_rng(seed)
        noisy_indices = rng.choice(n, size=n_noisy, replace=False) if n_noisy else np.array([], dtype=np.int64)

        # Wrong-label draw depends on noise_type. Uniform: rejection-free
        # shift-past-collision as before. Pair-flip: look up the fixed
        # confusable partner. A class outside the pair_map's keys (rare;
        # only if the caller passed a partial map) keeps its true label,
        # equivalent to that sample not being flipped.
        self._noisy = {}
        for idx in noisy_indices:
            true_label = self._raw_label(int(idx))
            if noise_type == "pairflip":
                if true_label not in self.pair_map:
                    continue    # partial map: leave the label untouched
                wrong = int(self.pair_map[true_label])
            else:
                wrong = int(rng.integers(0, num_classes - 1))
                if wrong >= true_label:
                    wrong += 1
            self._noisy[int(idx)] = wrong

    def _raw_label(self, idx: int) -> int:
        # Read the label WITHOUT running the base's transforms.
        #
        # Going through self.base[idx] would call CIFAR-10's __getitem__, which
        # runs RandomCrop + RandomHorizontalFlip. Each of those consumes torch's
        # global RNG, so with 20 noisy clients * ~500 samples this used to
        # burn ~10,000 draws before model init and quietly shift both model
        # weights and DataLoader shuffling relative to a no-noise run at the
        # same seed. The label itself does not need any of that.
        #
        # torchvision datasets expose the raw targets array; a Subset over one
        # forwards through .dataset. Fall back to __getitem__ only for
        # datasets that expose no such handle (e.g. the DummyDataset in tests
        # that has no .targets on the underlying object).
        base = self.base
        underlying = getattr(base, "dataset", None)
        indices = getattr(base, "indices", None)
        if underlying is not None and indices is not None:
            for attr in ("targets", "labels"):
                if hasattr(underlying, attr):
                    return int(getattr(underlying, attr)[int(indices[idx])])
        return int(base[idx][1])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        sample, label = self.base[idx]
        if idx in self._noisy:
            label = self._noisy[idx]
        return sample, label

    @property
    def num_noisy(self) -> int:
        return len(self._noisy)