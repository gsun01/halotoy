#!/bin/bash

nohup g++ -O3 -fopenmp simhalo.cpp && ./a.out &
wait
nohup python plot-halo.py &