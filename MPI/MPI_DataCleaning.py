# cleaning and monitoring process using MPI

from mpi4py import MPI
import pandas as pd
import re
import time
import psutil
import sys

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Read the CSV file only in the root process
filename = 'NoisyMobileDataLight.csv'
cleaned_file_name = 'cleaned_NoisyMobileDataLight.csv'
if rank == 0:
    df = pd.read_csv(filename)
    total_rows = df.shape[0]
    chunks = [df.iloc[i:i + 1000] for i in range(0, total_rows, 1000)]
else:
    chunks = None

# Broadcast the number of chunks
if rank == 0:
    num_chunks = len(chunks)
else:
    num_chunks = None

num_chunks = comm.bcast(num_chunks, root=0)

# Distribute chunks among processes
chunks_per_process = num_chunks // size
remainder = num_chunks % size

if rank == 0:
    distributed_chunks = []
    start_index = 0
    for i in range(size):
        end_index = start_index + chunks_per_process + (1 if i < remainder else 0)
        distributed_chunks.append(chunks[start_index:end_index])
        start_index = end_index
else:
    distributed_chunks = None

# Scatter the chunks
chunks = comm.scatter(distributed_chunks, root=0)

# Check if chunks is None or empty
if chunks is None or len(chunks) == 0:
    print(f"Process {rank} received no chunks to process.")
    cleaned_chunks = None
else:
    allowed_pattern = re.compile(r'[^a-zA-Z0-9+(),_ ,]')

    # Function to clean each cell
    def clean_cell(cell):
        if isinstance(cell, str):
            return allowed_pattern.sub('', cell)
        return cell

    process_start_time = time.time()
    print(f"Process {rank} starting processing.")

    cleaned_cells = 0
    try:
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            cleaned_chunk = chunk.apply(lambda col: col.map(clean_cell))
            cleaned_cells += (chunk != cleaned_chunk).sum().sum()
            print(f"Process {rank} completed chunk {i + 1} out of {len(chunks)}.")
            cleaned_chunks.append(cleaned_chunk)
        df_cleaned_chunk = pd.concat(cleaned_chunks)
    except Exception as e:
        print(f"Process {rank} encountered an error: {e}")
        df_cleaned_chunk = pd.DataFrame()

    process_end_time = time.time()
    processing_time = process_end_time - process_start_time
    memory_info = psutil.Process().memory_info()
    memory_usage = memory_info.rss / (1024 * 1024)  # Convert to MB

# Gather all cleaned chunks
cleaned_chunks = comm.gather(df_cleaned_chunk, root=0)

# Gather processing information
process_info = (processing_time, memory_usage, cleaned_cells)
all_process_info = comm.gather(process_info, root=0)

# Synchronize all processes before printing
comm.Barrier()

# Print processing information for all processes
if rank == 0:
    total_cleaned_cells = sum(info[2] for info in all_process_info)
    
    # Calculate total processing time based only on individual process cleaning times
    total_processing_time = sum(info[0] for info in all_process_info)  # This reflects only cleaning time

    # Calculate average throughput only after cleaning time has been calculated
    average_throughput = total_cleaned_cells / total_processing_time if total_processing_time > 0 else 0

    # Print individual processing times for each process
    print("\n**********************************************************")
    for i, info in enumerate(all_process_info):
        process_time, _, cleaned_cells = info
        print(f"*      Process {i} time taken: {process_time:.6f} sec                       *")
    
    df_cleaned = pd.concat(cleaned_chunks, ignore_index=True)
    df_cleaned.to_csv(cleaned_file_name, index=False)

    print("\n**********************************************************")
    print("*      Preprocessing Completed")
    print(f"*      Total cells processed: {total_cleaned_cells}")
    print(f"*      Total cleaning time : {total_processing_time:.6f} sec")  # Only cleaning time
    print(f"*      Average throughput: {int(average_throughput)} records/sec")
    print("**********************************************************")
else:
    print(f"Process {rank} completed its chunk.")