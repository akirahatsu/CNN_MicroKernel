import numpy as np
import time


# ==========================================
# SETTINGS
# ==========================================

n_kernel = 10
kernel_size = (3, 3)


# ==========================================
# INPUT
# ==========================================

# Same img used for PyTorch
x = np.asarray(img, dtype=np.float32)


# ==========================================
# CNN
# ==========================================

cnn = CNN(
    kernel_size=kernel_size,
    n_kernel=n_kernel
)


# ==========================================
# INITIALIZE PARAMETERS
# ==========================================

# First forward creates the kernel
output = cnn.forward(x)


# ==========================================
# WARM-UP
# ==========================================

for _ in range(10):

    output = cnn.forward(x)

    dl_dcnn = np.random.randn(
        *output.shape
    ).astype(np.float32)

    cnn.backward(dl_dcnn)


# ==========================================
# BENCHMARK
# Forward + Backward
# ==========================================

times = []

for _ in range(50):

    # Create upstream gradient outside timing
    output = cnn.forward(x)

    dl_dcnn = np.random.randn(
        *output.shape
    ).astype(np.float32)


    start = time.perf_counter()

    # --------------------------------------
    # Forward
    # --------------------------------------

    output = cnn.forward(x)

    # --------------------------------------
    # Backward
    # --------------------------------------

    cnn.backward(dl_dcnn)

    end = time.perf_counter()

    times.append(end - start)


# ==========================================
# RESULTS
# ==========================================

print("\nCNN_MicroKernel")
print("--------------------------------")
print("Device      : CPU")
print("Input shape :", x.shape)
print("Kernel      :", kernel_size)
print("N kernels   :", n_kernel)
print("Runs        :", len(times))
print("Average     :", sum(times) / len(times), "seconds")
print("Minimum     :", min(times), "seconds")
