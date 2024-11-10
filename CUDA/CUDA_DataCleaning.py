import time
from numba import cuda
import numpy as np
import pandas as pd
import warnings
from numba.core.errors import NumbaPerformanceWarning 

# Load dataset
file_path = 'NoisyMobileDataLight.csv'  # Update this to your file path
df = pd.read_csv(file_path)

# Columns to clean
columns_to_clean = [
    'Resolution', 'Display Type', 'Processor', 'Battery Type', 'Main Camera Resolution',
    'Front Camera Resolution', 'Video Recording Resolution', 'Flash Type',
    'Fingerprint Sensor', 'Build Material', 'Water & Dust Resistance Rating',
    'SIM Type', 'Speaker Type', 'Operating System', 'UI Skin', 'Voice Assistant',
    'Security Patch Frequency', '4G LTE Bands', 'Wi-Fi Version', 'Bluetooth Version',
    'USB Type', 'GPS'
]

# Ensure columns to clean exist in df and set them to string type, filling NaNs
df[columns_to_clean] = df[columns_to_clean].fillna('').astype(str)

# Convert each column into ASCII arrays (truncate or pad each to MAX_STR_LEN)
MAX_STR_LEN = 100
ascii_array_3d = np.array([
    [
        [ord(char) if 0 <= ord(char) < 128 else -1 for char in value[:MAX_STR_LEN]] + [-1] * (MAX_STR_LEN - len(value[:MAX_STR_LEN]))
        for value in df[col]
    ]
    for col in columns_to_clean
], dtype=np.int32)

# CUDA Constants
RECORDS_PER_THREAD = 15
THREADS_PER_BLOCK = 256
RECORDS_PER_BLOCK = THREADS_PER_BLOCK * RECORDS_PER_THREAD
total_records = ascii_array_3d.shape[1]

# CUDA Kernel for cleaning ASCII data
@cuda.jit
def clean_ascii_in_chunks(ascii_array_3d):
    idx = cuda.grid(1)  # Global thread index
    start, end = idx * RECORDS_PER_THREAD, min((idx + 1) * RECORDS_PER_THREAD, ascii_array_3d.shape[1])

    for col in range(ascii_array_3d.shape[0]):
        for i in range(start, end):
            pos = 0
            for j in range(ascii_array_3d.shape[2]):
                if 0 <= ascii_array_3d[col, i, j] < 128:
                    ascii_array_3d[col, i, pos] = ascii_array_3d[col, i, j]
                    pos += 1
            for j in range(pos, ascii_array_3d.shape[2]):
                ascii_array_3d[col, i, j] = -1

# Copy data to device
d_ascii_array_3d = cuda.to_device(ascii_array_3d)

# Calculate blocks per grid and start timer
blocks_per_grid = (total_records + RECORDS_PER_BLOCK - 1) // RECORDS_PER_BLOCK
print(f"Launching kernel with {blocks_per_grid} blocks and {THREADS_PER_BLOCK} threads per block.")
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
start_time = time.time()

# Launch kernel
clean_ascii_in_chunks[blocks_per_grid, THREADS_PER_BLOCK](d_ascii_array_3d)
cuda.synchronize()

# Calculate and print kernel execution time
kernel_execution_time = time.time() - start_time

# Copy cleaned data back to host
cleaned_ascii_array_3d = d_ascii_array_3d.copy_to_host()

# Convert cleaned ASCII arrays back to strings and update DataFrame
for col_idx, col_name in enumerate(columns_to_clean):
    df[col_name] = [''.join(chr(char_code) for char_code in row if char_code != -1).strip() for row in cleaned_ascii_array_3d[col_idx]]

# Display all cleaned data and kernel execution time
print("\nCleaned Data (All Records):")
print(df[columns_to_clean])

print(f"\nKernel execution time: {kernel_execution_time:.4f} seconds.")
