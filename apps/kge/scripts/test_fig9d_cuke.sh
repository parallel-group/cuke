#!/bin/bash

python test_cuke.py --dataset biokg --model RESCAL --batch_size 4096 --dim 512

python test_cuke.py --dataset biokg --model RESCAL --batch_size 4096 --dim 1024

python test_cuke.py --dataset biokg --model RESCAL --batch_size 8192 --dim 512

python test_cuke.py --dataset biokg --model RESCAL --batch_size 8192 --dim 1024