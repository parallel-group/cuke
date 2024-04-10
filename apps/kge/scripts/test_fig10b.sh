#!/bin/bash

python apps/kge/reproduce.py --dataset FB15k-237 --model TransR --batch_size 4096 --dim 512

python apps/kge/reproduce.py --dataset FB15k-237 --model TransR --batch_size 4096 --dim 1024

python apps/kge/reproduce.py --dataset FB15k-237 --model TransR --batch_size 8192 --dim 512

python apps/kge/reproduce.py --dataset FB15k-237 --model TransR --batch_size 8192 --dim 1024