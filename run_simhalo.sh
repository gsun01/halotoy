#!/bin/bash

g++ -O3 -fopenmp simhalo.cpp && ./a.out
python plot-halo.py