# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""

from test.unit.util import is_equal_tf

import numpy as np
import pytest
import tensorflow as tf

from deepreg.loss.util import (
    NegativeLossMixin,
    cauchy_kernel1d,
    gaussian_kernel1d_sigma,
    gaussian_kernel1d_size,
    rectangular_kernel1d,
    separable_filter,
    triangular_kernel1d,
)


@pytest.mark.parametrize("sigma", [1, 3, 2.2])
def test_cauchy_kernel1d(sigma):
    """
    Testing the 1-D cauchy kernel
    :param sigma: float
    :return:
    """
    tail = int(sigma * 5)
    expected = [1 / ((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)]
    expected = expected / np.sum(expected)
    got = cauchy_kernel1d(sigma)
    assert is_equal_tf(got, expected)


@pytest.mark.parametrize("sigma", [1, 3, 2.2])
def test_gaussian_kernel1d_sigma(sigma):
    """
    Testing the 1-D gaussian kernel given sigma as input
    :param sigma: float
    :return:
    """
    tail = int(sigma * 3)
    expected = [np.exp(-0.5 * x ** 2 / sigma ** 2) for x in range(-tail, tail + 1)]
    expected = expected / np.sum(expected)
    got = gaussian_kernel1d_sigma(sigma)
    assert is_equal_tf(got, expected)


@pytest.mark.parametrize("kernel_size", [3, 7, 11])
def test_gaussian_kernel1d_size(kernel_size):
    """
    Testing the 1-D gaussian kernel given size as input
    :param kernel_size: int
    :return:
    """
    mean = (kernel_size - 1) / 2.0
    sigma = kernel_size / 3

    grid = tf.range(0, kernel_size, dtype=tf.float32)
    expected = tf.exp(-tf.square(grid - mean) / (2 * sigma ** 2))

    got = gaussian_kernel1d_size(kernel_size)
    assert is_equal_tf(got, expected)


@pytest.mark.parametrize("kernel_size", [3, 7, 11])
def test_rectangular_kernel1d(kernel_size):
    """
    Testing the 1-D rectangular kernel
    :param kernel_size: int
    :return:
    """
    expected = tf.ones(shape=(kernel_size,), dtype=tf.float32)
    got = rectangular_kernel1d(kernel_size)
    assert is_equal_tf(got, expected)


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
def test_triangular_kernel1d(kernel_size):
    """
    Testing the 1-D triangular kernel
    :param kernel_size: int (odd number)
    :return:
    """
    expected = np.zeros(shape=(kernel_size,), dtype=np.float32)
    expected[kernel_size // 2] = kernel_size // 2 + 1
    for it_k in range(kernel_size // 2):
        expected[it_k] = it_k + 1
        expected[-it_k - 1] = it_k + 1

    got = triangular_kernel1d(kernel_size)
    assert is_equal_tf(got, expected)


def test_separable_filter():
    """
    Testing separable filter case where non
    zero length tensor is passed to the
    function.
    """
    k = np.ones((3, 3, 3, 3, 1), dtype=np.float32)
    array_eye = np.identity(3, dtype=np.float32)
    tensor_pred = np.zeros((3, 3, 3, 3, 1), dtype=np.float32)
    tensor_pred[:, :, 0, 0, 0] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)
    k = tf.convert_to_tensor(k, dtype=tf.float32)

    expect = np.ones((3, 3, 3, 3, 1), dtype=np.float32)
    expect = tf.convert_to_tensor(expect, dtype=tf.float32)

    get = separable_filter(tensor_pred, k)
    assert is_equal_tf(get, expect)


class MinusClass(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.name = "MinusClass"

    def call(self, y_true, y_pred):
        return y_true - y_pred


class MinusClassLoss(NegativeLossMixin, MinusClass):
    pass


@pytest.mark.parametrize("y_true,y_pred,expected", [(1, 2, 1), (2, 1, -1), (0, 0, 0)])
def test_negative_loss_mixin(y_true, y_pred, expected):
    """
    Testing NegativeLossMixin class that
    inverts the sign of any value
    returned by a function

    :param y_true: int
    :param y_pred: int
    :param expected: int
    :return:
    """

    y_true = tf.constant(y_true, dtype=tf.float32)
    y_pred = tf.constant(y_pred, dtype=tf.float32)

    got = MinusClassLoss().call(
        y_true,
        y_pred,
    )
    assert is_equal_tf(got, expected)
