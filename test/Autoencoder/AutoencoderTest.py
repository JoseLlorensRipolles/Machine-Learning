from src.Autoencoder.Autoencoder import Autoencoder
import src.Autoencoder.Autoencoder as AutoencoderFile

import unittest
import numpy as np


class TestAutoencoder(unittest.TestCase):

    def test_create_autoencoder(self):
        autoencoder = Autoencoder(50)

    def test_get_hidden_size(self):

        autoencoder = Autoencoder(10)
        self.assertEqual(autoencoder.get_hidden_layer_size(), 10)

        self.assertRaises(ValueError, autoencoder.__init__, 0)
        self.assertRaises(ValueError, autoencoder.__init__, -3)
        self.assertRaises(ValueError, autoencoder.__init__, -9000)

    def test_train(self):
        data = []
        for i in range(8):
            new_row = []
            for j in range(8):
                if i == j:
                    new_row.append(1)
                else:
                    new_row.append(-1)
            data.append(new_row)

        data = np.matrix(data)

        autoencoder = Autoencoder(3)
        autoencoder.train(data)

    def test_apply_activation(self):

        hidden_in = [-1, -0.5, 0, 0.5, 1]
        expected_hidden_out = [-0.7615941559557649, -0.46211715726000974, 0, 0.46211715726000974, 0.7615941559557649]
        np.testing.assert_array_equal(AutoencoderFile.apply_activation(hidden_in), expected_hidden_out)

        hidden_in = [[-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1]]
        expected_hidden_out = [[-0.7615941559557649, -0.46211715726000974, 0, 0.46211715726000974, 0.7615941559557649],\
                               [-0.7615941559557649, -0.46211715726000974, 0, 0.46211715726000974, 0.7615941559557649]]
        np.testing.assert_array_equal(AutoencoderFile.apply_activation(hidden_in), expected_hidden_out)


if __name__ == "__main__":
    unittest.main()