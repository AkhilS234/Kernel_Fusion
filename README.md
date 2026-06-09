# Kernel_Fusion
Kernel Fusion to minimize memory bandwidth bottlenecks

=== standard attention baseline ===
seq_len=  128 | latency=0.097ms | bandwidth=1.35 GB/s | HBM accessed=0.0001 GB
seq_len=  256 | latency=0.083ms | bandwidth=4.74 GB/s | HBM accessed=0.0004 GB
seq_len=  512 | latency=0.093ms | bandwidth=14.13 GB/s | HBM accessed=0.0013 GB
seq_len= 1024 | latency=0.109ms | bandwidth=43.38 GB/s | HBM accessed=0.0047 GB
seq_len= 2048 | latency=0.377ms | bandwidth=47.34 GB/s | HBM accessed=0.0178 GB
