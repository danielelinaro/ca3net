#!/bin/bash

inconfig="STDP_sym_single_a-thorny.json"
outconfig="tmp.json"

for sigma in 10 20 50 100 200 ; do
    for infile in spike_times_a-thorny_rate=*_N_thorny=8000.npz ; do
	sed -e 's/#SIGMA#/-'$sigma'/' $inconfig > $outconfig
        tmp="weights_${infile#spike_times_}"
	outfile="${tmp%.npz}_sigma=$sigma.npz"
	python3 evolve_weights_STDP.py -i $infile -o $outfile $outconfig
    done
done

