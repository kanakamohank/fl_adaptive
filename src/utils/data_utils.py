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


# CIFAR-10N label-set names, verbatim keys inside the human-annotated .pt file
# released by Wei et al. ICLR 2022 (https://github.com/UCSC-REAL/cifar-10-100n).
# Noise rates measured by the dataset authors against ground-truth clean_label:
#   clean         0% (sanity key; identical to vanilla CIFAR-10 labels)
#   aggre_label   9.03%   majority of 3 annotators
#   random_label1 17.23%  single-annotator rounds
#   random_label2 18.12%
#   random_label3 17.64%
#   worse_label   40.21%  worst single-annotator per image
_CIFAR10N_LABEL_SETS = (
    "clean", "aggre", "random1", "random2", "random3", "worst",
)
_CIFAR10N_KEY_MAP = {
    "clean":   "clean_label",
    "aggre":   "aggre_label",
    "random1": "random_label1",
    "random2": "random_label2",
    "random3": "random_label3",
    "worst":   "worse_label",
}


def load_cifar10n_labels(path: str, label_set: str) -> np.ndarray:
    """Return a numpy int64 array of length 50000 of human-annotator labels
    for the CIFAR-10 training set, keyed by global image index.

    Args:
        path: Path to CIFAR-10_human.pt downloaded from Wei et al. 2022
              (https://github.com/UCSC-REAL/cifar-10-100n). The file is a
              torch.load dict with keys clean_label, aggre_label, worse_label,
              random_label1, random_label2, random_label3.
        label_set: One of "clean", "aggre", "worst", "random1", "random2",
                   "random3". Determines which annotator stream to return.

    Raises:
        ValueError: unknown label_set, or the file is missing the expected
                    key, or the returned array is not length 50000.
    """
    if label_set not in _CIFAR10N_LABEL_SETS:
        raise ValueError(
            f"label_set must be one of {_CIFAR10N_LABEL_SETS}; got {label_set!r}"
        )
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions do not accept weights_only; fall back.
        blob = torch.load(path, map_location="cpu")
    key = _CIFAR10N_KEY_MAP[label_set]
    if key not in blob:
        raise ValueError(
            f"CIFAR-10N file at {path} missing key {key!r}; "
            f"got keys {sorted(blob.keys())}"
        )
    labels = blob[key]
    if hasattr(labels, "numpy"):
        labels = labels.numpy()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (50000,):
        raise ValueError(
            f"CIFAR-10N {key!r} must have shape (50000,); got {labels.shape}"
        )
    return labels


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

    # CIFAR-10 pair-flip default. Extends the canonical Patrini et al. 2017
    # asymmetric map (`bird->airplane, deer->horse, cat<->dog, truck->automobile`)
    # to a SYMMETRIC map that covers all 10 classes:
    #   airplane <-> bird           canonical pair, both fly on sky background
    #   automobile <-> truck        canonical pair, wheeled vehicles
    #   cat <-> dog                 canonical pair, four-legged pets
    #   deer <-> horse              canonical pair, four-legged large mammals
    #   frog <-> ship               EXTENSION (not from literature); chosen so
    #                               every class has a partner and the effective
    #                               noise rate stays uniform across classes
    # Callers reproducing Patrini's exact asymmetric benchmark should pass their
    # own 4-entry map via pair_map; the constructor will then reject partial
    # coverage, so if you use the asymmetric benchmark you must exclude
    # samples of the unmapped classes from the noisy set.
    _CIFAR10_PAIRFLIP = {0: 2, 2: 0, 1: 9, 9: 1, 3: 5, 5: 3, 4: 7, 7: 4, 6: 8, 8: 6}

    def __init__(self, base: Dataset, noise_fraction: float,
                 num_classes: int = 10, seed: int = 0,
                 noise_type: str = "uniform",
                 pair_map: dict = None,
                 cifar10n_labels: "np.ndarray" = None):
        if not (0.0 <= noise_fraction <= 1.0):
            raise ValueError(f"noise_fraction must be in [0, 1]; got {noise_fraction}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2; got {num_classes}")
        if noise_type not in ("uniform", "pairflip", "cifar10n"):
            raise ValueError(
                f"noise_type must be 'uniform', 'pairflip', or 'cifar10n'; "
                f"got {noise_type!r}"
            )

        self.base = base
        self.num_classes = num_classes
        self.noise_type = noise_type

        # CIFAR-10N: real-human mislabels from Wei et al. ICLR 2022. Instead
        # of synthesizing wrong labels, we swap in the human-annotator label
        # for the noisy sample. The caller passes an array of length
        # `len(underlying_cifar10)` (typically 50000 for the training set),
        # one human label per image index into the FULL CIFAR-10 dataset.
        # NoisyLabelSubset wraps a torch.utils.data.Subset; we read the
        # underlying global index via base.indices to look up the human label.
        #
        # Why this is not collusive like synthetic pair-flip: different human
        # workers made different mistakes, so two noisy clients labelled by
        # (say) worker-random1 do NOT share a fixed flip map. The resulting
        # per-client gradients do not point in a shared wrong direction, which
        # is what killed BVD under pair-flip at 30% flip rate.
        if noise_type == "cifar10n":
            if cifar10n_labels is None:
                raise ValueError(
                    "noise_type='cifar10n' requires cifar10n_labels=<array of "
                    "human labels indexed by global CIFAR-10 image index>"
                )
            if not hasattr(cifar10n_labels, "__len__"):
                raise ValueError(
                    "cifar10n_labels must be a 1-D array-like of ints"
                )
            self.cifar10n_labels = cifar10n_labels
        else:
            self.cifar10n_labels = None

        # Pair-flip map. Default to the CIFAR-10 pattern; callers with a
        # different class count or a different pairing must pass one in.
        if noise_type == "pairflip":
            resolved = pair_map if pair_map is not None else self._CIFAR10_PAIRFLIP
            # Sanity-check the map: keys and values must be valid class
            # ids, no self-loops, AND full coverage over range(num_classes).
            #
            # Partial coverage was previously permitted with a silent
            # `continue` in the flip loop below -- that let samples of
            # unmapped classes keep their true label, silently reducing
            # the effective noise rate below what noise_fraction advertised.
            # For an asymmetric benchmark (e.g. Patrini's 4-entry map), the
            # caller must exclude samples of the unmapped classes from the
            # noisy pool themselves; the wrapper refuses partial coverage.
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
            missing = set(range(num_classes)) - set(resolved.keys())
            if missing:
                raise ValueError(
                    f"pair_map missing entries for classes {sorted(missing)}; "
                    f"full coverage over range({num_classes}) is required so "
                    f"the effective noise rate matches noise_fraction. For an "
                    f"asymmetric benchmark, restrict noise to samples whose "
                    f"class IS in pair_map's keys and pass that subset."
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
        # pair_map is now validated to cover every class in range(num_classes),
        # so the pair-flip branch always finds a partner. No silent skips.
        # self._noisy maps subset-local idx -> wrong label, and len(self._noisy)
        # is counted as num_noisy. Semantics across all three noise types:
        # an entry here means the sample's final label DIFFERS from ground
        # truth. Uniform and pairflip guarantee a strict wrong label by
        # construction; cifar10n only registers an entry when the human label
        # actually disagrees with ground truth (no-op swaps where the
        # annotator happened to get it right are skipped). That keeps
        # num_noisy == "effective flips" and keeps parity with the other
        # noise types for downstream analyses and paper tables.
        #
        # Follow-ons that need "how many samples the cifar10n path visited"
        # (as opposed to flipped) can read num_cifar10n_candidates below.
        self._noisy = {}
        skipped_cifar10n_agreements = 0
        for idx in noisy_indices:
            true_label = self._raw_label(int(idx))
            if noise_type == "pairflip":
                wrong = int(self.pair_map[true_label])
            elif noise_type == "cifar10n":
                # Resolve this subset-local idx to the global CIFAR-10 index.
                # NoisyLabelSubset is typically wrapped around a Subset; the
                # Subset's .indices maps subset-local to dataset-global. If
                # base has no .indices (plain full dataset or custom wrapper
                # exposing targets directly), use idx itself as global.
                global_idx = (int(getattr(self.base, "indices", range(len(self.base)))[int(idx)]))
                wrong = int(self.cifar10n_labels[global_idx])
                if wrong == true_label:
                    # Human annotator happened to agree with ground truth
                    # for this sample. Not a flip. Skip the entry so
                    # num_noisy stays an honest flip count.
                    skipped_cifar10n_agreements += 1
                    continue
            else:
                wrong = int(rng.integers(0, num_classes - 1))
                if wrong >= true_label:
                    wrong += 1
            self._noisy[int(idx)] = wrong
        # Exposed so callers can tell the difference between "the wrapper
        # visited N candidate samples" and "N samples were effectively
        # noisy." For non-cifar10n paths these are always equal.
        self.num_cifar10n_candidates = len(noisy_indices)
        self.num_cifar10n_agreements_skipped = skipped_cifar10n_agreements

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