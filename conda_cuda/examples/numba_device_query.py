import numba
from numba import cuda
import numpy as np
import math

@cuda.jit
def simple_kernel(buffer):
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        buffer[0] = math.gamma(0.1)

# pylint:disable=unsubscriptable-object
def launch_kernel(buffer):
    simple_kernel[1, 1](buffer)

if __name__ == "__main__":
    print("numba version", numba.__version__)
    cuda.detect()
    devices = cuda.list_devices()
    if  len(devices) > 0:
        print("Testing a simple kernel in device 0")
        cuda.select_device(0)
        host_buffer = np.zeros(1, dtype=np.float32)
        device_buffer = cuda.to_device(host_buffer)
        launch_kernel(device_buffer)
        host_buffer2 = device_buffer.copy_to_host()
        assert np.isclose(host_buffer2[0], math.gamma(0.1))
        print("Testing successful!")
