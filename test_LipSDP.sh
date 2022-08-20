#!/bin/sh

cd LipSDP/LipSDP
python solve_sdp.py --form layer --weight-path ../../data/weights/random_weights.mat
