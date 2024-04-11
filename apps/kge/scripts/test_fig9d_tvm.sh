#!/bin/bash

python apps/kge/test_tvm.py --dataset biokg --model RESCAL --batch_size 4096 --dim 512

python apps/kge/test_tvm.py --dataset biokg --model RESCAL --batch_size 4096 --dim 1024

python apps/kge/test_tvm.py --dataset biokg --model RESCAL --batch_size 8192 --dim 512

python apps/kge/test_tvm.py --dataset biokg --model RESCAL --batch_size 8192 --dim 1024