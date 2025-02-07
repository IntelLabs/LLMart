#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from datasets import Dataset

from llmart.data import _split_dataset


@pytest.fixture
def ds3():
    return Dataset.from_dict({"name": ["a", "b", "c"]})


@pytest.fixture
def ds2():
    return Dataset.from_dict({"name": ["d", "e"]})


@pytest.fixture
def ds1():
    return Dataset.from_dict({"name": ["f"]})


def test_split_dataset(ds3):
    train, minitrain, val, test = _split_dataset(ds3)

    assert len(train) == len(ds3)
    assert len(minitrain) == len(ds3)
    assert len(val) == 0
    assert len(test) == len(ds3)


def test_split_dataset_train_test(ds3, ds2):
    train, minitrain, val, test = _split_dataset(
        train=ds3, val=ds2, n_train=1, n_test=2
    )

    assert len(train) == 1
    assert len(minitrain) == 1
    assert len(val) == len(val)
    assert len(test) == 2


def test_split_dataset_train_val_test(ds3):
    train, minitrain, val, test = _split_dataset(
        train=ds3, n_train=1, n_val=1, n_test=1
    )

    assert len(train) == 1
    assert len(minitrain) == 1
    assert len(val) == 1
    assert len(test) == 1


def test_split_dataset_minitrain(ds3):
    train, minitrain, val, test = _split_dataset(train=ds3, n_minitrain=0)

    assert len(train) == len(ds3)
    assert len(minitrain) == 0
    assert len(val) == 0
    assert len(test) == len(ds3)


def test_split_dataset_fails(ds3):
    with pytest.raises(AttributeError):
        _split_dataset(train=ds3, n_val=1)


def test_split_dataset_with_val(ds3, ds2):
    train, minitrain, val, test = _split_dataset(train=ds3, val=ds2)

    assert len(train) == len(train)
    assert len(minitrain) == len(train)
    assert len(val) == len(val)
    assert len(test) == 3


def test_split_dataset_with_test(ds3, ds1):
    train, minitrain, val, test = _split_dataset(train=ds3, test=ds1)

    assert len(train) == len(ds3)
    assert len(minitrain) == len(ds3)
    assert len(val) == 0
    assert len(test) == len(ds1)


def test_split_dataset_with_val_and_test(ds3, ds2, ds1):
    train, minitrain, val, test = _split_dataset(train=ds3, val=ds2, test=ds1)

    assert len(train) == len(ds3)
    assert len(minitrain) == len(ds3)
    assert len(val) == len(ds2)
    assert len(test) == len(ds1)


def test_split_dataset_val_with_test(ds3, ds1):
    train, minitrain, val, test = _split_dataset(train=ds3, test=ds1, n_val=1)

    assert len(train) == len(ds3) - 1
    assert len(minitrain) == len(ds3) - 1
    assert len(val) == 1
    assert len(test) == len(ds1)


def test_split_dataset_test_with_test(ds3, ds1):
    train, minitrain, val, test = _split_dataset(train=ds3, test=ds1, n_test=2)

    assert len(train) == len(ds3)
    assert len(minitrain) == len(ds3)
    assert len(val) == 0
    assert len(test) == len(ds1)
