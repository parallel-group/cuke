#!/bin/bash

python apps/kge/test.py --dataset FB15k-237 --model TransH --batch_size 4096 --dim 512

python apps/kge/test.py --dataset FB15k-237 --model TransH --batch_size 4096 --dim 1024

python apps/kge/test.py --dataset FB15k-237 --model TransH --batch_size 8192 --dim 512

python apps/kge/test.py --dataset FB15k-237 --model TransH --batch_size 8192 --dim 1024