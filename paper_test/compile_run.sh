
nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a -o bin/W_probe W_probe.cu -lcudart
nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a -o bin/fp16_lossless fp16_lossless.cu -lcudart
nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a -o bin/largest_k_instr largest_k_instr.cu -lcudart


./bin/largest_k_instr
./bin/fp16_lossless
./bin/W_probe
