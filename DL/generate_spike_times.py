# -*- coding: utf-8 -*-

import os
import sys
import json
from tqdm import tqdm
import numpy
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import matplotlib.pyplot as plt

from utils import *

prog_name = os.path.basename(sys.argv[0])

def pos(t, speed, length):
    return (speed * t) % length


def time_constant(x, middle_PF, sigma_PF):
    return np.exp(-(x - middle_PF)**2 / (2 * (sigma_PF/(2*np.pi)*track_length)**2))


def firing_rate(t, max_rate, length_PF, start_PF, middle_PF, sigma_PF, animal_speed, track_length, ftheta):
    x = pos(t, animal_speed, track_length)
    rate = max_rate * time_constant(x, middle_PF, sigma_PF) * \
                np.cos(2 * np.pi * ftheta * t + \
                       np.pi / length_PF * (x - start_PF))
    if np.isscalar(t):
        return rate if rate > 0 else 0
    rate[rate < 0] = 0
    return rate

DEBUG = False

def usage():
    print(f'usage: {prog_name} [<options>] <config_file>')
    print( '')
    print( '    -o, --output   output file name')
    print( '    -f, --force    force overwrite of existing data file')
    print( '    --plots        generate plots for a random subset of cells')
    print( '    -h, --help     print this help message and exit')
    print( '')
    
if __name__ == '__main__':

    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        usage()
        sys.exit(0)

    i = 1
    n_args = len(sys.argv)

    output_file = 'spike_times.npz'
    make_plots = False
    n_place_cells_to_plot = 5
    force = False
    
    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-o', '--output'):
            output_file = sys.argv[i+1]
            i += 1
        elif arg in ('-f', '--force'):
            force = True
        elif arg == '--plots':
            make_plots = True
        else:
            break
        i += 1
        
    if i == n_args:
        usage()
        sys.exit(1)

    if os.path.isfile(output_file) and not force:
        print(f'{prog_name}: output file "{output_file}" exists: use -f to overwrite')
        sys.exit(1)

    config_file = sys.argv[i]
    if not os.path.isfile(config_file):
        print(f'{prog_name}: {config_file}: no such file.')
        sys.exit(2)

    config = json.load(open(config_file, 'r'))

    track_length      = config['track_length']        # [cm]
    animal_speed      = config['animal_speed']        # [cm/s]
    ftheta            = config['theta_frequency']     # [Hz]
    total_time        = config['total_time']          # [s]
    n_neurons         = config['n_neurons']
    max_rate          = config['max_rate']            # [spike/s]
    baseline_rate     = config['baseline_rate']       # [Hz]
    length_PF         = config['place_field_length']  # [cm]
    PC_ratio          = config['place_cell_ratio']
    refractory_period = config['refractory_period']   # [s]
    lap_time          = track_length / animal_speed   # [s]
    #r = track_length / (2 * np.pi)

    seed = config['seed'] if 'seed' in config else None
    if seed == None:
        import time
        seed = int(time.time() * 1000)
    master_rs = RandomState(MT19937(SeedSequence(seed)))
    print('Seed:', seed)

    cell_types = n_neurons.keys()
    middle_PF, start_PF, sigma_PF, phi_PF_rad = {}, {}, {}, {}
    seeds = {}
    place_cell = {}
    spikes = {}
    for cell_type in cell_types:
        middle_PF[cell_type] = np.sort(track_length * master_rs.uniform(size=n_neurons[cell_type]))
        start_PF[cell_type] = (middle_PF[cell_type] - length_PF[cell_type] / 2) % track_length
        sigma_PF[cell_type] = length_PF[cell_type] / 2 / track_length * 2 * np.pi / 3
        phi_PF_rad[cell_type] = length_PF[cell_type] / (2 * np.pi)
        seeds[cell_type] = master_rs.randint(1000000, size=n_neurons[cell_type])
        place_cell[cell_type] = master_rs.uniform(size=n_neurons[cell_type]) < PC_ratio[cell_type]
        place_cells_to_plot = np.random.permutation(np.where(place_cell[cell_type])[0])
        if place_cells_to_plot.size > n_place_cells_to_plot:
            place_cells_to_plot = place_cells_to_plot[:n_place_cells_to_plot]
        random_states = [RandomState(MT19937(SeedSequence(seed))) for seed in seeds[cell_type]]
        tarp = refractory_period[cell_type]
        spikes[cell_type] = []
        print(f'Generating spike times for {cell_type} cell type:')
        for neuron_id in tqdm(range(n_neurons[cell_type])):
            rs = random_states[neuron_id]
            if place_cell[cell_type][neuron_id]:
                start = start_PF[cell_type][neuron_id]
                middle = middle_PF[cell_type][neuron_id]
                rate_fun = lambda t: firing_rate(t, max_rate[cell_type], length_PF[cell_type],
                                                 start, middle, sigma_PF[cell_type],
                                                 animal_speed, track_length, ftheta)
                spks,all_spks = make_inhomogeneous_poisson_spike_train(rate_fun,
                                                                       max_rate[cell_type],
                                                                       tend=total_time,
                                                                       random_state=rs,
                                                                       refractory_period=tarp,
                                                                       full_output=True)
            else:
                spks = make_poisson_spike_train(baseline_rate[cell_type], tend=total_time,
                                                random_state=rs, refractory_period=tarp)
                all_spks = spks

            spikes[cell_type].append(spks)

            if make_plots and neuron_id in place_cells_to_plot:
                pos_fun = lambda times: [pos(t, animal_speed, track_length) for t in times]
                edges = np.r_[0 : total_time : lap_time]
                idx = np.digitize(all_spks, edges)
                all_spks_reshaped = [all_spks[idx == i+1] - i * lap_time for i in range(idx.max())]
                if place_cell[cell_type][neuron_id]:
                    idx = np.digitize(spks, edges)
                    spks_reshaped = [spks[idx == i+1] - i * lap_time for i in range(idx.max())]
                
                fig,ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
                max_laps = 50
                rasterplot(pos_fun(all_spks_reshaped), max_laps, ax=ax[0])
                nu, edges, count = psth(all_spks_reshaped, binwidth=0.1 / ftheta, interval=[0, lap_time])
                ax[1].plot(pos_fun(edges), nu, 'k', lw=1, label='Hom Poisson rate')
                if place_cell[cell_type][neuron_id]:
                    rasterplot(pos_fun(spks_reshaped), max_laps, ax=ax[0], marker='x', color='r', markersize=3)
                    nu, edges, count = psth(spks_reshaped, binwidth=0.1 / ftheta, interval=[0, lap_time])
                    ax[1].plot(pos_fun(edges), nu, 'm', lw=1, label='Inhom Poisson rate')
                    ax[1].plot(pos_fun(edges), rate_fun(edges), 'g', lw=1, label='Theor rate')
                    ylim = ax[1].get_ylim()
                    ax[1].plot(start + np.zeros(2), ylim, 'c--', lw=1)
                    ax[1].plot(middle + np.zeros(2), ylim, 'c--', lw=1)
                    ax[1].plot(middle + length_PF[cell_type] / 2 + np.zeros(2), ylim, 'c--', lw=1)
                ax[0].set_xlim([0, track_length])
                ax[0].set_ylim([0, np.min([max_laps, len(all_spks_reshaped)])+1])
                ax[0].set_ylabel('Lap #')
                ax[1].set_ylabel('Firing rate (spikes/s)')
                ax[1].set_xlabel('Position (cm)')
                ax[1].legend(loc='best')
                for a in ax:
                    for side in 'right','top':
                        a.spines[side].set_visible(False)
                fig.tight_layout()
                plt.savefig(f'{cell_type}_{neuron_id:05d}.pdf')


    np.savez_compressed(output_file, spike_trains=spikes, seed=seed, seeds=seeds, place_cell=place_cell, config=config)

