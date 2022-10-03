from scipy.io import loadmat
import numpy as np
import torch

# dataset loader
def load_dataset(
    dataset_path: str,
    random_split: int = 0,
    subsample_train: float = 0.0,
):
    # load dataset
    print("dataset_path:", str(dataset_path))
    data = torch.from_numpy(loadmat(dataset_path)["data"]).float()
    print("data.shape:", data.shape)

    # shuffle
    N = data.shape[0]
    np.random.seed(random_split)
    data = data[np.random.permutation(np.arange(N)), :]

    # make train/test split
    n_train_val = int(0.8 * N)
    n_train = int(0.8 * n_train_val)

    train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
    val_x, val_y = data[n_train:n_train_val, :-1], data[n_train:n_train_val, -1]
    test_x, test_y = data[n_train_val:, :-1], data[n_train_val:, -1]

    # # normalize features
    # mean = train_x.mean(dim=-2, keepdim=True)
    # std = train_x.std(dim=-2, keepdim=True) + 1e-6
    # train_x = (train_x - mean) / std
    # val_x = (val_x - mean) / std
    # test_x = (test_x - mean) / std

    # normalize features
    # mean = train_x.mean(dim=-2, keepdim=True)
    # std = train_x.std(dim=-2, keepdim=True) * 8 + 1e-6
    # train_x = (train_x - mean) / std + 0.5
    # val_x = (val_x - mean) / std + 0.5
    # test_x = (test_x - mean) / std + 0.5
    min_x = torch.min(train_x, axis=0, keepdim=True)[0]
    max_x = torch.max(train_x, axis=0, keepdim=True)[0]
    train_x = (train_x - min_x) / (max_x - min_x + 1e-6)
    val_x = (val_x - min_x) / (max_x - min_x + 1e-6)
    test_x = (test_x - min_x) / (max_x - min_x + 1e-6)

    # normalize labels
    mean, std = train_y.mean(), train_y.std()
    train_y = (train_y - mean) / std
    val_y = (val_y - mean) / std
    test_y = (test_y - mean) / std

    # subsample train, if requested
    if subsample_train > 0.0:
        n_train = np.int(np.ceil(subsample_train * n_train))
        train_x, train_y = train_x[:n_train, :], train_y[:n_train]

    # make continguous
    train_x, train_y = train_x.contiguous(), train_y.contiguous()
    val_x, val_y = val_x.contiguous(), val_y.contiguous()
    test_x, test_y = test_x.contiguous(), test_y.contiguous()

    print("train_x.shape:", train_x.shape)
    print("full kernel size: %.1f GB" % ((train_x.shape[0] ** 2) * 4 / (1024 ** 3)))
    print(
        f"train_x: min {train_x.min():.3f}, max {train_x.max():.3f}, mean {train_x.mean():.3f}, std {train_x.std():.3f}"
    )
    print(
        f"val_x: min {val_x.min():.3f}, max {val_x.max():.3f}, mean {val_x.mean():.3f}, std {val_x.std():.3f}"
    )
    print(
        f"test_x: min {test_x.min():.3f}, max {test_x.max():.3f}, mean {test_x.mean():.3f}, std {test_x.std():.3f}"
    )
    print(
        f"train_y: min {train_y.min():.3f}, max {train_y.max():.3f}, mean {train_y.mean():.3f}, std {train_y.std():.3f}"
    )
    print(
        f"val_y: min {val_y.min():.3f}, max {val_y.max():.3f}, mean {val_y.mean():.3f}, std {val_y.std():.3f}"
    )
    print(
        f"test_y: min {test_y.min():.3f}, max {test_y.max():.3f}, mean {test_y.mean():.3f}, std {test_y.std():.3f}"
    )
    print()
    return train_x, train_y, val_x, val_y, test_x, test_y
