import torch
import time

# ==========================================
# SETTINGS
# ==========================================

n_kernel = 10
kernel_size = (3, 3)

# Same input used by your CNN
x = torch.tensor(img, dtype=torch.float32)

batch, channel, height, width = x.shape


# ==========================================
# CNN
# ==========================================

cnn = torch.nn.Conv2d(
    in_channels=channel,
    out_channels=n_kernel,
    kernel_size=kernel_size,
    stride=1,
    padding=0,
    bias=True
)


# ==========================================
# WARM-UP
# ==========================================

for _ in range(10):

    cnn.zero_grad(set_to_none=True)

    output = cnn(x)

    dl_dcnn = torch.randn_like(output)

    output.backward(dl_dcnn)


# ==========================================
# BENCHMARK
# Forward + Backward
# ==========================================

times = []

for _ in range(50):

    cnn.zero_grad(set_to_none=True)

    # Create upstream gradient before timing
    output = cnn(x)
    dl_dcnn = torch.randn_like(output)

    start = time.perf_counter()

    # Forward
    output = cnn(x)

    # Backward
    output.backward(dl_dcnn)

    end = time.perf_counter()

    times.append(end - start)


# ==========================================
# RESULTS
# ==========================================

print("\nPyTorch CPU")
print("--------------------------------")
print("Input shape :", tuple(x.shape))
print("Kernel      :", kernel_size)
print("N kernels   :", n_kernel)
print("Runs        :", len(times))
print("Average     :", sum(times) / len(times), "seconds")
print("Minimum     :", min(times), "seconds")
