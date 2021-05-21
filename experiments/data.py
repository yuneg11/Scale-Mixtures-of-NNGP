import os
import math
import urllib
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow_datasets as tfds
from sklearn.datasets import load_boston


regression_datasets = [
    "boston", "concrete", "energy", "kin8nm",
    "naval", "plant", "wine-red", "wine-white", "yacht",
]

classification_datasets = [
    "mnist", "iris", "test_cls",
]

datasets = regression_datasets + classification_datasets

dataset_urls = {
    "concrete": {
        "Concrete_Data.xls": "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "Concrete_Readme.txt": "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Readme.txt",
    },
    "energy": {
        "ENB2012_data.xlsx": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
    },
    "kin8nm": {
        "dataset_2175_kin8nm.csv": "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv",
    },
    "naval": {
        "UCI CBM Dataset.zip": "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
    },
    "plant": {
        "CCPP.zip": "http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
    },
    "wine": {
        "winequality-red.csv": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "winequality-white.csv": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "winequality.names": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names",
    },
    "yacht": {
        "yacht_hydrodynamics.data": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
    },
}


def _urlretrieve(url, filename, chunk_size = 1024):
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def _download_url(url, filepath):
    try:
        print("Download {} to {}".format(url, filepath))
        _urlretrieve(url, filepath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + filepath)
            _urlretrieve(url, filepath)
        else:
            raise e


def _extract_zip(filepath):
    to_path = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath, 'r') as z:
        z.extractall(to_path)
    # os.remove(filepath)  # Remove ZIP file after extraction


def _download_dataset(name, root):
    root = os.path.expanduser(root)
    dataset_path = os.path.join(root, name)
    os.makedirs(dataset_path, exist_ok=True)

    files = dataset_urls[name]

    for filename, url in files.items():
        filepath = os.path.join(dataset_path, filename)

        if not os.path.isfile(filepath):
            _download_url(url, filepath)

            if filename.endswith(".zip"):
                _extract_zip(filepath)


# Regression

def get_regression_dataset(name, root="./data", y_newaxis=True):
    if name == "boston":  # Boston Housing
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
        x, y = load_boston(return_X_y=True)

    elif name == "concrete":  # Concrete Compressive Strength
        # http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
        _download_dataset(name, root)

        filepath = os.path.join(root, "concrete/Concrete_Data.xls")
        excel_data = pd.read_excel(filepath)
        data = excel_data.to_numpy()

        x, y = data[:, :8], data[:, 8]

    elif name == "energy":  # Energy efficiency
        # http://archive.ics.uci.edu/ml/datasets/Energy+efficiency
        _download_dataset(name, root)

        filepath = os.path.join(root, "energy/ENB2012_data.xlsx")
        excel_data = pd.read_excel(filepath)
        data = excel_data.to_numpy()

        x, y = data[:, :8], data[:, 8]  # TODO: need to check

    elif name == "kin8nm":  # kin8nm
        # https://www.openml.org/d/189
        _download_dataset(name, root)

        filepath = os.path.join(root, "kin8nm/dataset_2175_kin8nm.csv")
        csv_data = pd.read_csv(filepath)
        data = csv_data.to_numpy()

        x, y = data[:, :8], data[:, 8]

    elif name == "naval":  # Condition Based Maintenance of Naval Propulsion Plants
        # http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
        _download_dataset(name, root)

        filepath = os.path.join(root, "naval/UCI CBM Dataset/data.txt")
        txt_data = pd.read_table(filepath, delim_whitespace=" ")
        data = txt_data.to_numpy()

        x, y = data[:, :16], data[:, 16]

    elif name == "plant":  # Combined Cycle Power Plant
        # http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
        _download_dataset(name, root)

        filepath = os.path.join(root, "plant/CCPP/Folds5x2_pp.xlsx")
        excel_data = pd.read_excel(filepath)
        data = excel_data.to_numpy()

        x, y = data[:, :4], data[:, 4]

    elif name == "wine-red" or name == "wine-white":  # Wine Quality
        # http://archive.ics.uci.edu/ml/datasets/Wine+Quality
        _download_dataset("wine", root)

        if name == "wine-red":
            filepath = os.path.join(root, "wine/winequality-red.csv")
        else:
            filepath = os.path.join(root, "wine/winequality-white.csv")

        csv_data = pd.read_csv(filepath, delimiter=";")
        data = csv_data.to_numpy()

        x, y = data[:, :11], data[:, 11]

    elif name == "yacht":  # Yacht Hydrodynamics
        # http://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics
        _download_dataset(name, root)

        filepath = os.path.join(root, "yacht/yacht_hydrodynamics.data")
        txt_data = pd.read_table(filepath, delim_whitespace=" ")
        data = txt_data.to_numpy()

        x, y = data[:, :6], data[:, 6]

    else:
        raise KeyError("Unsupported dataset '{}'".format(name))

    if y_newaxis:
        y = y[:, np.newaxis]

    return x, y


def split_dataset(
    x, y,
    train, valid, test,
    normalize_x=True, normalize_y=True
):
    fractions = train + valid + test

    if not math.isclose(fractions, 1.0) and fractions > 1.0:
        raise ValueError("Sum of fractions exceed 1.0")

    train_num = int(train * len(x))
    x_train = x[:train_num]
    y_train = y[:train_num]

    valid_num = int(valid * len(x))
    x_valid = x[train_num: train_num + valid_num]
    y_valid = y[train_num: train_num + valid_num]

    if math.isclose(fractions, 1.0):
        x_test = x[train_num + valid_num:]
        y_test = y[train_num + valid_num:]
    elif fractions < 1.0:
        test_num = int(test * len(x))
        x_test = x[train_num + valid_num: train_num + valid_num + test_num]
        y_test = y[train_num + valid_num: train_num + valid_num + test_num]

    if normalize_x:
        x_std = np.std(x_train, axis=0)
        x_mean = np.mean(x_train, axis=0)

        x_train = (x_train - x_mean) / x_std
        x_valid = (x_valid - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std

        np.nan_to_num(x_train, copy=False)
        np.nan_to_num(x_valid, copy=False)
        np.nan_to_num(x_test, copy=False)

    if normalize_y:
        y_std = np.std(y_train, axis=0)
        y_mean = np.mean(y_train, axis=0)

        y_train = (y_train - y_mean) / y_std
        y_valid = (y_valid - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# Classification

def permute_dataset(x, y, seed=0):
    idx = np.random.RandomState(seed).permutation(x.shape[0])
    permuted_x = x[idx]
    permuted_y = y[idx]
    return permuted_x, permuted_y


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def get_classification_dataset(name, root="./data", train_num=None, test_num=None, seed=0):
    if name != "test_cls":
        ds_builder = tfds.builder(name)

    if name == "mnist":
        ds_train, ds_test = tfds.as_numpy(
            tfds.load(
                name,
                split=["train" + ("[:%d]" % train_num if train_num is not None else ""),
                    "test" + ("[:%d]" % test_num if test_num is not None else "")],
                batch_size=-1,
                as_dataset_kwargs={"shuffle_files": False},
                data_dir=root,
            )
        )
        dataset = (ds_train["image"], ds_train["label"], ds_test["image"], ds_test["label"])
        x_train, y_train, x_test, y_test = dataset

        num_classes = ds_builder.info.features["label"].num_classes
        
        
        # DEBUG START -> Move to top of return
        # y_train_0 = jnp.sum(y_train[:, [0, 2, 4, 6, 8]], axis=1, keepdims=True)
        # y_train_1 = jnp.sum(y_train[:, [1, 3, 5, 7, 9]], axis=1, keepdims=True)
        # y_train = jnp.concatenate([y_train_0, y_train_1], axis=1)
        # DEBUG END

    elif name == "iris":
        ds_train, = tfds.as_numpy(
            tfds.load(
                name,
                # split=["train" + ("[:%d]" % train_num if train_num is not None else "")],
                split=["train"],
                batch_size=-1,
                as_dataset_kwargs={"shuffle_files": False},
                data_dir=root,
            )
        )
        x, y = ds_train["features"], ds_train["label"]
        x, y = permute_dataset(x, y, seed=109)
        x_train, y_train, x_test, y_test = x[:train_num], y[:train_num], x[train_num:], y[train_num:]

        num_classes = ds_builder.info.features["label"].num_classes

    elif name == "test_cls":
        func = lambda x: np.sin(x * 3 * np.pi) + 0.3 * np.cos(x * 9 * np.pi) + 0.5 * np.sin(x * 7 * np.pi)

        n = train_num + test_num

        rng = np.random.RandomState(123)
        # min_x, max_x = -np.pi, +np.pi
        min_x, max_x = -1, +1
        x = np.expand_dims(np.linspace(min_x, max_x, n, endpoint=False), axis=1)
        y = func(x) + 0.2 * rng.randn(n, 1)

        x_train, y_train, x_test, y_test = x[:train_num], y[:train_num], x[train_num:], y[train_num:]

        # x_train = (x_train - np.mean(x_train)) / np.std(x_train)
        # x_test = (x_test - np.mean(x_train)) / np.std(x_train)

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        return x_train, y_train, x_test, y_test

    else:
        raise KeyError("Unsupported dataset '{}'".format(name))

    y_train = _one_hot(y_train, num_classes)
    y_test = _one_hot(y_test, num_classes)

    x_train, y_train = permute_dataset(x_train, y_train, seed=seed)

    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    x_test = (x_test - np.mean(x_train)) / np.std(x_train)

    return x_train, y_train, x_test, y_test


def get_dataset(name, root="./data", **kwargs):
    if name in regression_datasets:
        return get_regression_dataset(name, root=root, **kwargs)
    elif name in classification_datasets:
        return get_classification_dataset(name, root=root, **kwargs)
    else:
        raise KeyError("Unsupported dataset '{}'".format(name))
