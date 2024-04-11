#!/bin/bash

python test_tvm.py --dataset wn18 --model neg_TransF --batch_size 4096 --dim 512

python test_tvm.py --dataset wn18 --model neg_TransF --batch_size 4096 --dim 1024

python test_tvm.py --dataset wn18 --model neg_TransF --batch_size 8192 --dim 512

python test_tvm.py --dataset wn18 --model neg_TransF --batch_size 8192 --dim 1024