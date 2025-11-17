nvcc -O3 -std=c++17 -arch=sm_120a \
-I/home/xtzhao/cutlass/include \
-I/home/xtzhao/cutlass/include/ \
-I/home/xtzhao/cutlass/tools/util/include/ \
-I/home/xtzhao/cutlass/examples/common/ \
-I/home/xtzhao/cutlass/include/cute \
-o bin/fp8_acc_bitwidth fp8_acc_bitwidth.cu -lcudart

./bin/fp8_acc_bitwidth