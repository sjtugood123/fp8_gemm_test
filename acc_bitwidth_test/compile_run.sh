# nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a \
# -I/home/xtzhao/cutlass/include \
# -I/home/xtzhao/cutlass/tools/util/include/ \
# -I/home/xtzhao/cutlass/examples/common/ \
# -I/home/xtzhao/cutlass/include/cute \
# -o bin/fp8_acc_bitwidth fp8_acc_bitwidth.cu -lcudart

# nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a \
# -I/home/xtzhao/cutlass/include \
# -I/home/xtzhao/cutlass/tools/util/include/ \
# -I/home/xtzhao/cutlass/examples/common/ \
# -I/home/xtzhao/cutlass/include/cute \
# -o bin/fp4_acc_bitwidth fp4_acc_bitwidth.cu -lcudart

# nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a \
# -I/home/xtzhao/cutlass/include \
# -I/home/xtzhao/cutlass/tools/util/include/ \
# -I/home/xtzhao/cutlass/examples/common/ \
# -I/home/xtzhao/cutlass/include/cute \
# -o bin/test_mxfp4 test_mxfp4.cu -lcudart


# nvcc -diag-suppress 550 -O3 -std=c++17 -arch=sm_120a \
# -I/home/xtzhao/cutlass/include \
# -I/home/xtzhao/cutlass/tools/util/include/ \
# -I/home/xtzhao/cutlass/examples/common/ \
# -I/home/xtzhao/cutlass/include/cute \
# -o bin/fp4_block_scalevec1 fp4_block_scalevec1.cu -lcudart

nvcc -diag-suppress 550 -diag-suppress 177 -O3 -std=c++17 -arch=sm_120a \
-I/home/xtzhao/cutlass/include \
-I/home/xtzhao/cutlass/tools/util/include/ \
-I/home/xtzhao/cutlass/examples/common/ \
-I/home/xtzhao/cutlass/include/cute \
-o bin/fp4_block_scalevec2 fp4_block_scalevec2.cu -lcudart

# ./bin/fp8_acc_bitwidth
# ./bin/fp4_acc_bitwidth
# ./bin/test_mxfp4
# ./bin/fp4_block_scalevec1
./bin/fp4_block_scalevec2

# cuobjdump -sass ./bin/fp8_acc_bitwidth > ./sass/fp8.sass
# cuobjdump -sass ./bin/fp4_acc_bitwidth > ./sass/fp4.sass
# cuobjdump -sass ./bin/test_mxfp4 > ./sass/test_mxfp4.sass
# cuobjdump -sass ./bin/fp4_block_scalevec1 > ./sass/fp4_block_scalevec1.sass
# cuobjdump -sass ./bin/fp4_block_scalevec2 > ./sass/fp4_block_scalevec2.sass