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

result:

On 5090 tensor core, the acc bitwidth is 25 for both fp4 and fp8
