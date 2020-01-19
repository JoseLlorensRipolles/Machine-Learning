import unittest
import numpy as np
from src.Regression.HousePricesKaggle.HousePricesDataset import *

class TestHousePricesRegression(unittest.TestCase):

    def test_dataset_creation(self):
        matrix = np.array([["Id","MSSubClass"],
                          ["1", "50"],
                          ["2", "60"],
                          ["3", "50"]])

        new_matrix = one_hot_encode(matrix, cols_to_encode=[1])
        print(new_matrix)
