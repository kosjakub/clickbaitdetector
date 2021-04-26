import unittest
import clickbaitdetector.clickbaitdetector as a
"""
Run with PYTHONPATH=. python tests/test_dummy.py
"""


class TestDummy(unittest.TestCase):

    def test_fun(self):
        
        
        p = a.check_clickbait("Is this France’s favourite pastry?", a.default_coefficients)
        if p >= 0.5:
            p = 1
        elif p < 0.5:
            p = 0
        self.assertEqual(p, 1)

        p = a.check_clickbait("Huge chunks of ice fall from glacier", a.default_coefficients)
        if p >= 0.5:
            p = 1
        elif p < 0.5:
            p = 0
        self.assertEqual(p, 0)

        p = a.check_clickbait("How I took my family on the run for 19 years", a.default_coefficients)
        if p >= 0.5:
            p = 1
        elif p < 0.5:
            p = 0
        self.assertEqual(p, 1)
        
     


if __name__ == '__main__':
    unittest.main()