#!/bin/bash

# nohup g++ -O3 -fopenmp simhalo.cpp && ./a.out $1 $2 $3 $4 &
nohup g++ -O3 -fopenmp simhalo.cpp -o b17v1.out && ./b17v1.out 0.151 17 1.5 1.0 &