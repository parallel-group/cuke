#!/bin/bash

python test_cuke.py --dataset FB15k-237 --model TransH --batch_size 4096 --dim 512

python test_cuke.py --dataset FB15k-237 --model TransH --batch_size 4096 --dim 1024

python test_cuke.py --dataset FB15k-237 --model TransH --batch_size 8192 --dim 512

python test_cuke.py --dataset FB15k-237 --model TransH --batch_size 8192 --dim 1024