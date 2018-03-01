import LabAsignments.Lab1.Part1_2 as TestedObject
import unittest
import numpy as np


class Test(unittest.TestCase):

    def test_phi_function(self):
        self.assertEqual(TestedObject.phi([1]), [0.46211715726000979])
        self.assertEqual(TestedObject.phi([0]), [0])
        self.assertEqual(TestedObject.phi([-1]), [-0.46211715726000979])
        self.assertAlmostEqual(TestedObject.phi([10]), [1], delta=0.01)
        self.assertAlmostEqual(TestedObject.phi([-10]), [-1], delta=0.01)
        self.assertTrue(np.array_equal(TestedObject.phi([[0, 1], [1, 0]]), [[0, 0.46211715726000979], [0.46211715726000979, 0]]))

    def test_phiprime_function(self):
        self.assertTrue(np.array_equal(TestedObject.phiprime([0, 0.8]), [0.46211715726000979, 0]))


    def test_summed_input_signal_function(self):
        W = [[1, 2, 3], [0, 3, 2]]
        patterns = [[1, 1], [1, 3], [1, 7]]
        expected_summed_input = [[6, 28], [5, 23]]
        self.assertTrue(np.array_equal(TestedObject.calculate_summed_input_signal(W, patterns), expected_summed_input))



if __name__ == '__main__':
    unittest.main()