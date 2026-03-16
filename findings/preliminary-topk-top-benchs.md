# Preliminary top-k/top-p benchmarks

top-k/top-p disabled

bsz=1
                                      Provider  median_ms  min_ms  max_ms  iters
0                                 fused-triton      1.418   1.417   1.420     48
1                               naive-compiled      1.490   1.488   1.494     66
2  flashinfer:top_k_top_p_sampling_from_logits      1.648   1.647   1.651     59

bsz=256
                                      Provider  median_ms  min_ms  max_ms  iters
0                                 fused-triton      5.489   5.441   5.491     16
2  flashinfer:top_k_top_p_sampling_from_logits      6.276   6.266   6.280     15
1                               naive-compiled      6.644   6.610   6.649     15

top-k/top-p enabled

bsz=1
                                      Provider  median_ms  min_ms  max_ms  iters
1                               naive-compiled      1.559   1.558   1.562     63
2  flashinfer:top_k_top_p_sampling_from_logits      1.641   1.639   1.643     59
0                                 fused-triton      1.651   1.635   1.668     59

bsz=256
                                      Provider  median_ms  min_ms  max_ms  iters
2  flashinfer:top_k_top_p_sampling_from_logits      6.562   6.549   6.694     15
1                               naive-compiled      7.734   7.725   7.740     12
0                                 fused-triton     14.294  14.230  14.304      6

Summary:

- The fused-triton kernel with active top-k/top-p degrades very badly from 5.489ms to 14.294ms at bsz=256.
- In comparison top_k_top_p_sampling_from_logits stays approximately equal from 6.276ms to 6.562ms at bsz=256.
- The current top-p/top-k implementation is not optimized in the fused-triton kernel.