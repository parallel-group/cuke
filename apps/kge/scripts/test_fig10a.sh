#!/bin/bash

python apps/kge/reproduce.py --dataset wn18 --model neg_TransF --batch_size 4096 --dim 512

python apps/kge/reproduce.py --dataset wn18 --model neg_TransF --batch_size 4096 --dim 1024

python apps/kge/reproduce.py --dataset wn18 --model neg_TransF --batch_size 8192 --dim 512

python apps/kge/reproduce.py --dataset wn18 --model neg_TransF --batch_size 8192 --dim 1024