from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from lib.utils import load_imdb
from lib.lr.config import LogisticRegressionConfig
import torch, numpy as np


def get_data_loaders(
    config: LogisticRegressionConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, TfidfVectorizer]:
    # load raw splits (train, validation, test)
    train_ds, val_ds, test_ds = load_imdb()

    # instantiate TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
    )

    # fit on train, transform all splits
    X_train = vectorizer.fit_transform(train_ds["text"])
    X_val = vectorizer.transform(val_ds["text"])
    X_test = vectorizer.transform(test_ds["text"])

    y_train = np.array(train_ds["label"])
    y_val = np.array(val_ds["label"])
    y_test = np.array(test_ds["label"])

    # convert to PyTorch TensorDataset
    train_dataset = TensorDataset(
        torch.from_numpy(X_train.toarray()).float(), torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val.toarray()).float(), torch.from_numpy(y_val).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test.toarray()).float(), torch.from_numpy(y_test).long()
    )

    # create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vectorizer
