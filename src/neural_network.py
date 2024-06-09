import numpy as np
from numba import cuda, float32

# Define CUDA kernels
@cuda.jit
def relu(x, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = max(0, x[i])

@cuda.jit
def softmax(x, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        row = x[i]
        row_max = float32(np.max(row))
        row_exp = np.exp(row - row_max)
        out[i] = row_exp / float32(np.sum(row_exp))

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

class NeuralNetwork():
    '''init NN with ReLu'''
    def __init__(self, width_array: np.array):
        self.layers = []
        assert len(width_array) >= 2
        for i in range(len(width_array) - 1):
            xavier_normal_std = (np.sqrt(2 / (width_array[i] + width_array[i + 1])))
            layer = np.random.normal(loc = xavier_normal_std, size= (width_array[i], width_array[i + 1])).astype(np.float32)
            self.layers.append(layer)

    def foreward(self, x: np.array) -> np.array:
        for i in range(self.num_layers):
            weights_gpu = cuda.to_device(self.weights[i])
            #biases_gpu = cuda.to_device(self.biases[i])

            layer_output = np.empty_like(cuda.to_device(X.astype(np.float32) @ self.weights[i]))
            matrix_multiply(activations[i], weights_gpu, layer_output)
            relu(layer_output + biases_gpu, layer_output)
            activations.append(layer_output)

        output_layer = activations[-1]
        softmax(output_layer, output_layer)

        x_device = cuda.to_device(x)
        output_device = cuda.to_device(np.zeros(self.layers[-1].shape[-1]))

        threads_per_block = 32
        blocks_per_grid = (x.shape[0] + threads_per_block - 1) // threads_per_block
        

    @cuda.jit
    def forward_kernel():
        pass

    def __str__(self):
        description = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, np.ndarray):  # Assuming layers are numpy arrays representing weights
                description.append(f'Layer {i + 1}: {layer.shape[0]} neurons, connected to {layer.shape[1]} neurons')
        return '\n'.join(description)