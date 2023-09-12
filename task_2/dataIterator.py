from torchtext.data import Iterator, BucketIterator
import torch

batch_size = 32


def iterator_construct_train(data, device):
    train_data, dev_data = data.split([0.8, 0.2])
    data_iterator, dev_iterator = Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=device,
        shuffle=True
    )
    return data_iterator, dev_iterator


def iterator_construct_test(data, device):
    data_iterator = Iterator(
        data,
        batch_size=batch_size,
        device=device,
        sort=False,
        sort_within_batch=False,
        repeat=False,
        shuffle=False
    )
    return data_iterator
