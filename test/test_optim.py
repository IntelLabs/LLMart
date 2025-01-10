#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import copy
import torch
from torch.testing import make_tensor, assert_close

from llmart import GreedyCoordinateGradient, Coordinate


def closure():
    return torch.randn(1).squeeze()


def make_onehot(*args, **kwargs):
    tensor = make_tensor(*args, **kwargs)
    index = tensor.argmax(dim=-1)

    tensor.data[:] = 0.0
    tensor.data.scatter_(-1, index[..., None], 1.0)

    return tensor


def test_optim():
    param = make_onehot(20, 512, dtype=torch.float, device="cpu", requires_grad=True)
    optim = GreedyCoordinateGradient([param])

    with pytest.raises(ValueError):
        optim.coordinate_replacements


def test_optim_param_shape():
    param = make_onehot(1, 20, 512, dtype=torch.float, device="cpu", requires_grad=True)

    with pytest.raises(ValueError):
        GreedyCoordinateGradient([param])


def test_optim_multi_params_fails():
    param1 = make_onehot(20, 512, dtype=torch.float, device="cpu", requires_grad=True)
    param2 = make_onehot(20, 512, dtype=torch.float, device="cpu", requires_grad=True)

    with pytest.raises(ValueError):
        GreedyCoordinateGradient([param1, param2])


def test_optim_param_groups_fails():
    param1 = dict(
        params=make_onehot(
            20, 512, dtype=torch.float, device="cpu", requires_grad=True
        ),
        negative_only=False,
    )
    param2 = dict(
        params=make_onehot(
            20, 512, dtype=torch.float, device="cpu", requires_grad=True
        ),
        negative_only=True,
    )

    with pytest.raises(ValueError):
        GreedyCoordinateGradient([param1, param2])


def test_optim_wrong_shape():
    param = make_onehot(512, dtype=torch.float, device="cpu", requires_grad=True)

    with pytest.raises(ValueError):
        GreedyCoordinateGradient([param])


@pytest.fixture
def params():
    param = make_onehot(
        3, 512, dtype=torch.float, device="cpu", requires_grad=True, low=-1, high=1
    )

    # manually set argmax tokens
    param.data[:] = 0
    param.data[0, 1] = 1.0
    param.data[1, 2] = 1.0
    param.data[2, 3] = 1.0
    assert param[0].argmax(-1) == 1
    assert param[1].argmax(-1) == 2
    assert param[2].argmax(-1) == 3
    assert param[0].max() == 1.0
    assert param[1].max() == 1.0
    assert param[2].max() == 1.0

    param.grad = make_tensor(
        3, 512, dtype=torch.float, device="cpu", low=-1, high=1
    ).abs()
    assert (param.grad > 0).all()

    # manually set argmin grads
    param.grad[0, 123] = -2.0
    param.grad[1, 321] = -3.0
    param.grad[2, 213] = -4.0
    assert param.grad[0].argmin() == 123
    assert param.grad[1].argmin() == 321
    assert param.grad[2].argmin() == 213
    assert param.grad[0].min() == -2.0
    assert param.grad[1].min() == -3.0
    assert param.grad[2].min() == -4.0

    return [param]


def test_optim_coordinate_replacements(params):
    optim = GreedyCoordinateGradient(params, n_swaps=3)

    replacements = optim.coordinate_replacements
    assert replacements is not None
    assert len(replacements) == 3

    replacements2 = optim.coordinate_replacements
    assert replacements == replacements2


def test_optim_coordinate_replacements_set(params):
    optim = GreedyCoordinateGradient(params)

    replacements = optim.coordinate_replacements
    optim.coordinate_replacements = replacements[:1]

    replacements2 = optim.coordinate_replacements
    assert replacements[:1] == replacements2

    replacements3 = optim.coordinate_replacements
    assert replacements2 == replacements3


def test_optim_coordinate_replacements_negative_only(params):
    optim = GreedyCoordinateGradient(
        params,
        negative_only=True,
        coord_randk=0,
        coord_topk=0,
        global_topk=0,
        n_tokens=1,
        n_swaps=100,
        ignored_values=None,
    )

    coord_0 = Coordinate({0: 123})
    assert coord_0 in optim.coordinate_replacements

    coord_1 = Coordinate({1: 321})
    assert coord_1 in optim.coordinate_replacements

    coord_2 = Coordinate({2: 213})
    assert coord_2 in optim.coordinate_replacements


def test_optim_coordinate_replacements_negative_only_with_all_positive_grad():
    param = make_onehot(20, 512, dtype=torch.float, device="cpu", requires_grad=True)
    optim = GreedyCoordinateGradient(
        [param],
        negative_only=True,
        coord_randk=0,
        coord_topk=0,
        global_topk=0,
        n_swaps=512,
    )

    # Make gradients all positive on all token positions
    param.grad = make_tensor(20, 512, dtype=torch.float, device="cpu").abs()
    assert (param.grad >= 0).all()

    with pytest.warns(
        UserWarning,
        match="No tokens can be swapped at all! Check self._negative_only = True!",
    ):
        assert optim.coordinate_replacements == []
        with pytest.warns(
            UserWarning, match="All 512 attempted swaps do not re-encode!"
        ):
            # Take the step
            previous_data = copy.deepcopy(param.data)
            optim.step(closure)

    assert torch.all(param.data == previous_data)  # Step did nothing


def test_optim_coordinate_replacements_negative_only_with_many_positive_grad():
    param = make_onehot(20, 512, dtype=torch.float, device="cpu", requires_grad=True)
    optim = GreedyCoordinateGradient(
        [param],
        negative_only=True,
        coord_randk=0,
        coord_topk=0,
        global_topk=0,
        n_tokens=10,
    )

    # Make gradients all positive, except for a single token position
    param.grad = make_tensor(20, 512, dtype=torch.float, device="cpu").abs()
    param.grad[0, ...] = -param.grad[0, ...]
    assert (param.grad[1:, ...] >= 0).all()
    assert (param.grad[0, ...] < 0).all()

    with pytest.warns(
        UserWarning,
        match=r"Fewer than 10 token\(s\) survived, will instead swap tuples of 1 token\(s\)!",
    ):
        replacements = optim.coordinate_replacements
        # Check that exactly one (and matching) token will be swapped
        swapped_tokens = set([index for repl in replacements for index in repl.index])

        assert len(swapped_tokens) == 1
        assert list(swapped_tokens)[0] == 0


def test_optim_coordinate_replacements_ignored_values(params):
    optim = GreedyCoordinateGradient(
        params,
        negative_only=True,
        coord_randk=0,
        coord_topk=0,
        global_topk=0,
        n_tokens=1,
        n_swaps=100,
        ignored_values=torch.tensor([123], dtype=torch.long),
    )

    replacements = optim.coordinate_replacements
    assert len(replacements) == 2  # Only two unique swaps exist

    # We ignore token value 123 so 0th coordinate should not be updated
    coord_0 = Coordinate({0: 123})
    assert coord_0 not in optim.coordinate_replacements

    coord_1 = Coordinate({1: 321})
    assert coord_1 in optim.coordinate_replacements

    coord_2 = Coordinate({2: 213})
    assert coord_2 in optim.coordinate_replacements


def test_optim_coordinate_replacements_n_tokens(params):
    optim = GreedyCoordinateGradient(
        params,
        negative_only=True,
        coord_randk=0,
        coord_topk=0,
        global_topk=0,
        n_tokens=2,
        n_swaps=100,
    )

    replacements = optim.coordinate_replacements
    assert len(replacements) == 3  # Only (3 choose 2) = 3 unique swaps exist

    # get sorted indices of replacements
    replacements = [sorted(r.index) for r in optim.coordinate_replacements]

    # There should be (len(params[0])=3 choose n_tokens=2)=3 multi-token sets
    assert len({frozenset(r) for r in replacements}) == 3


def test_optim_coordinate_replacements_coord_topk(params):
    optim = GreedyCoordinateGradient(
        params,
        negative_only=False,
        coord_randk=0,
        coord_topk=1,
        global_topk=0,
        n_tokens=1,
        n_swaps=100,
    )

    coord_0 = Coordinate({0: 123})
    assert coord_0 in optim.coordinate_replacements

    coord_1 = Coordinate({1: 321})
    assert coord_1 in optim.coordinate_replacements

    coord_2 = Coordinate({2: 213})
    assert coord_2 in optim.coordinate_replacements


def test_optim_coordinate_replacements_global_topk(params):
    optim = GreedyCoordinateGradient(
        params,
        negative_only=False,
        coord_randk=0,
        coord_topk=0,
        global_topk=3,
        n_tokens=1,
        n_swaps=100,
    )

    coord_0 = Coordinate({0: 123})
    assert coord_0 in optim.coordinate_replacements

    coord_1 = Coordinate({1: 321})
    assert coord_1 in optim.coordinate_replacements

    coord_2 = Coordinate({2: 213})
    assert coord_2 in optim.coordinate_replacements


def test_optim_all_non_reencoding(params):
    def closure_inf():
        return torch.tensor([torch.inf])

    optim = GreedyCoordinateGradient(params, n_swaps=1024)

    # Save a copy so we can compare later
    previous_data = copy.deepcopy(params[0].data)

    # Take the step
    with pytest.warns(UserWarning, match="All 1024 attempted swaps do not re-encode!"):
        optim.step(closure_inf)

    # Make sure step did nothing
    assert_close(params[0].data, previous_data)

    # Take  a step and make sure it did something
    optim.step(closure)
    assert not torch.allclose(params[0].data, previous_data)


def test_optim_generator(params):
    def generator():
        param_losses = []
        while True:
            param_idx = yield []
            if param_idx is None:
                yield param_losses
                break
            param_losses.append((param_idx, closure()))

    optim = GreedyCoordinateGradient(params, n_swaps=1024)

    # Save a copy so we can compare later
    previous_data = copy.deepcopy(params[0].data)

    # Take  a step and make sure it did something
    optim.step(generator)
    assert not torch.allclose(params[0].data, previous_data)


def test_optim_generator_empty(params):
    def generator():
        while True:
            _ = yield []

    optim = GreedyCoordinateGradient(params, n_swaps=1024)

    # Save a copy so we can compare later
    previous_data = copy.deepcopy(params[0].data)

    # Take the step
    with pytest.warns(UserWarning, match="All 1024 attempted swaps do not re-encode!"):
        optim.step(generator)

    # Make sure step did nothing
    assert_close(params[0].data, previous_data)


def test_optim_generator_inf(params):
    def generator():
        param_losses = []
        while True:
            param_idx = yield []
            if param_idx is None:
                yield param_losses
                break
            param_losses.append((param_idx, torch.tensor(torch.inf)))

    optim = GreedyCoordinateGradient(params, n_tokens=3, n_swaps=1024)

    # Save a copy so we can compare later
    previous_data = copy.deepcopy(params[0].data)

    # Take the step
    with pytest.warns(UserWarning, match="All 1024 attempted swaps do not re-encode!"):
        optim.step(generator)

    # Make sure step did nothing
    assert_close(params[0].data, previous_data)


def test_optim_uniques(params):
    optim = GreedyCoordinateGradient(
        params,
        negative_only=False,
        coord_randk=0,
        coord_topk=4,
        global_topk=0,
        n_tokens=3,
        n_swaps=10000,
    )

    replacements = optim.coordinate_replacements
    # Only 4 ** 3 = 64 unique replacements exist
    assert len(replacements) == 64

    optim = GreedyCoordinateGradient(
        params,
        negative_only=False,
        coord_randk=0,
        coord_topk=6,
        global_topk=0,
        n_tokens=2,
        n_swaps=10000,
    )

    replacements = optim.coordinate_replacements
    # Only 6 ** 2 * (3 choose 2 = 3) = 108 unique replacements exist
    assert len(replacements) == 108
