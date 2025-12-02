gemm:

```
./refresh
python example.py
```

bitwidth test:

```
chmod +x ./acc_bitwidth_test/compile_run.sh
./acc_bitwidth_test/compile_run.sh
```

bitwidth test with triton:
Triton should be 3.6.0

```
git clone https://github.com/triton-lang/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .# takes about 40 minutes
cd ~/fp8_gemm/acc_bitwidth_test
python triton_fp4.py
```

result:

On 5090 tensor core, the acc bitwidth is 25 for both fp4 and fp8

The acc bitwidth is 29 for fp4 using triton.
