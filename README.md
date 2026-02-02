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
cd ~/fp8_gemm_test/acc_bitwidth_test
python fp16_probe_triton.py
```

FPRev

```
pip install graphviz
git clone https://github.com/peichenxie/FPRev.git
cd FPRev
pip install .
pip install -r experiments/requirements.txt
python acc_order/try_emu.py
```


result:

5090/QMMA the acc bitwidth is 25 for fp4,fp8,fp16

5090/(triton,OMMA)the acc bitwidth is 29 for fp4
