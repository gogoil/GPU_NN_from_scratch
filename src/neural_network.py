import numpy as np
from numba import cuda, float32
def Neural_network():
    '''init NN with ReLu'''
    def __init__(self, width_array: np.array):
        self.layers = []
        assert len(width_array) >= 2
        for i in range(len(width_array) - 1):
            xavier_normal_std = (np.sqrt(2 / (width_array[i] + width_array[i + 1])))
            layer = np.random.normal((loc = std, size= (width_array[i], width_array[i + 1])))
            self.layers.append(layer)


    # def init_weights(self):
    # '''Xavier init'''
    #     for layer in 

    def foreward(self, x: np.array) -> np.array:
        pass

    @cuda.jit
    def forward_kernel():
        pass