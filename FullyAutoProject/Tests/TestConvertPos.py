import logging
import unittest

import numpy as np
import torch

def convert_pos(pos):
    """
    Converts a position from a Tensor or list to a tuple of integers.

    Parameters:
    - pos: A Tensor, list, or tuple containing two numeric values.

    Returns:
    - A tuple of two integers representing the (x, y) position.
    """
    if isinstance(pos, torch.Tensor):
        if pos.dim() == 1 and pos.numel() == 2:
            converted = (int(round(pos[0].item())), int(round(pos[1].item())))
            logging.debug(f"convert_pos: Tensor input {pos.tolist()} converted to {converted}")
            return converted
        else:
            raise ValueError(f"Expected a 1D Tensor with 2 elements, got shape {pos.shape}")
    elif isinstance(pos, (list, tuple, np.ndarray)):
        if len(pos) == 2:
            converted = (int(round(pos[0])), int(round(pos[1])))
            logging.debug(f"convert_pos: list/tuple/ndarray input {pos} converted to {converted}")
            return converted
        else:
            raise ValueError(f"Expected a list/tuple/ndarray with 2 elements, got {len(pos)} elements")
    else:
        raise TypeError(f"Unsupported type for position conversion: {type(pos)}")

class TestConvertPos(unittest.TestCase):
    def test_tensor_valid(self):
        pos = torch.tensor([4.7, 3.2])
        expected = (5, 3)
        self.assertEqual(convert_pos(pos), expected)

    def test_tensor_invalid_shape(self):
        pos = torch.tensor([4.7, 3.2, 1.0])
        with self.assertRaises(ValueError):
            convert_pos(pos)

    def test_list_valid(self):
        pos = [4.7, 3.2]
        expected = (5, 3)
        self.assertEqual(convert_pos(pos), expected)

    def test_tuple_invalid_length(self):
        pos = (4.7,)
        with self.assertRaises(ValueError):
            convert_pos(pos)

    def test_invalid_type(self):
        pos = "invalid"
        with self.assertRaises(TypeError):
            convert_pos(pos)

    def test_numpy_array_valid(self):
        pos = np.array([4.7, 3.2])
        expected = (5, 3)
        self.assertEqual(convert_pos(pos), expected)

    def test_numpy_array_invalid_length(self):
        pos = np.array([4.7, 3.2, 1.0])
        with self.assertRaises(ValueError):
            convert_pos(pos)

if __name__ == '__main__':
    unittest.main()
