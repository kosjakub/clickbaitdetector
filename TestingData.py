import numpy as np

class TestingData:
    def __init__(self, h, v, p):
        self.headline = h
        self.features_vector = v
        self.propability = p
    def toString(self):
    	return self.headline + '^' + np.array_str(self.features_vector) + '^' + str(self.propability)