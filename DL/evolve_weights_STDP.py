# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from scipy import sparse
from brian2 import *
import platform
if platform.system() == 'Linux':
    set_device('cpp_standalone')

prog_name = os.path.basename(sys.argv[0])

def usage():
    print(f'usage: {prog_name} [<options>] <config_file>')
    print( '')
    print( '    -i, --spike-times  file containing input spike times')
    print( '    -o, --output       output file name')
    print( '    -f, --force        force overwrite of existing data file')
    print( '    -h, --help         print this help message and exit')
    print( '')
    
if __name__ == '__main__':

    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        usage()
        sys.exit(0)

    k = 1
    n_args = len(sys.argv)

    output_file = None
    spike_times_file = None
    force = False
    
    while k < n_args:
        arg = sys.argv[k]
        if arg in ('-o', '--output'):
            output_file = sys.argv[k+1]
            k += 1
        elif arg in ('-f', '--force'):
            force = True
        elif arg in ('-i', '--spike-times'):
            spike_times_file = sys.argv[k+1]
            k += 1
        else:
            break
        k += 1
        
    if k == n_args:
        usage()
        sys.exit(1)

    config_file = sys.argv[k]
    if not os.path.isfile(config_file):
        print(f'{prog_name}: {config_file}: no such file.')
        sys.exit(2)

    config = json.load(open(config_file, 'r'))

    # STDP parameters (see `optimization/analyse_STDP.py`)
    taup         = config['taup'] * ms
    taum         = config['taum'] * ms
    Ap           = config['Ap']
    Am           = config['Am']
    wmax         = config['wmax']
    scale_factor = config['scale_factor']
    w_init       = config['w_init']
    t_end        = config['t_end'] if 't_end' in config else None
    # needed to reproduce Brian1 results
    Ap *= wmax
    Am *= wmax

    if spike_times_file is None:
        spike_times_file = config['spike_times_file']
    if not os.path.isfile(spike_times_file):
        print(f'{prog_name}: {spike_times_file}: no such file.')
        sys.exit(3)

    if output_file is None:
        if 'spike_times' in spike_times_file:
            output_file = spike_times_file.replace('spike_times', 'weights')
        else:
            output_file = 'synaptic_weights.npz'
    if os.path.isfile(output_file) and not force:
        print(f'{prog_name}: output file "{output_file}" exists: use -f to overwrite')
        sys.exit(1)

    data = np.load(spike_times_file, allow_pickle=True)
    spike_times_config = data['config'].item()
    place_cell = data['place_cell'].item()
    spike_trains = data['spike_trains'].item()
    cell_types = list(spike_trains.keys())
    if t_end is None:
        t_end = np.ceil(max([max(map(lambda s: s[-1], spks)) for spks in spike_trains.values()]))
    
    n_cells = {}
    spike_times = {}
    neuron_indices = {}
    spike_gen_group = {}
    for cell_type in cell_types:
        n_cells[cell_type] = len(spike_trains[cell_type])
        spike_times[cell_type] = np.concatenate(spike_trains[cell_type])
        neuron_indices[cell_type] = np.concatenate([i*np.ones_like(spks, dtype=np.int32) for i,spks in enumerate(spike_trains[cell_type])])
        idx = np.argsort(spike_times[cell_type])
        spike_times[cell_type] = spike_times[cell_type][idx]
        neuron_indices[cell_type] = neuron_indices[cell_type][idx]
        spike_gen_group[cell_type] = SpikeGeneratorGroup(n_cells[cell_type], neuron_indices[cell_type], spike_times[cell_type] * second)

    connections = config['connectivity']

    synapses = {key: {} for key in set(connections['pre'])}
    has_conv_input = {key: {} for key in set(connections['pre'])}
    rnd = np.random.uniform
    if 'conv' not in connections:
        n_pairs = len(connections['pre'])
        connections['conv'] = [None for _ in range(n_pairs)]
        connections['sigma'] = [None for _ in range(n_pairs)]
        connections['n_pre'] = [None for _ in range(n_pairs)]
    for pre,post,prob,conv,sigma,n_pre in zip(connections['pre'],
                                              connections['post'],
                                              connections['prob'],
                                              connections['conv'],
                                              connections['sigma'],
                                              connections['n_pre']):
        synapses[pre][post] = Synapses(spike_gen_group[pre], spike_gen_group[post],
                                       """
                                       w : 1
                                       dA_presyn/dt = -A_presyn/taup : 1 (event-driven)
                                       dA_postsyn/dt = -A_postsyn/taum : 1 (event-driven)
                                       """,
                                       on_pre="""
                                       A_presyn += Ap
                                       w = clip(w + A_postsyn, 0, wmax)
                                       """,
                                       on_post="""
                                       A_postsyn += Am
                                       w = clip(w + A_presyn, 0, wmax)
                                       """,
                                       name=f'{pre.replace("-","")}_to_{post.replace("-","")}')
        has_conv_input[pre][post] = np.zeros(n_cells[post], dtype=bool)
        if conv is not None:
            print(f'{pre} -> {post} has convergent inputs.')
            has_conv_input[pre][post] = rnd(size=n_cells[post]) <= conv
            x = np.arange(n_cells[pre])
            pc_ratio = spike_times_config['place_cell_ratio'][pre]
            for idx_post in range(n_cells[post]):
                if has_conv_input[pre][post][idx_post]:
                    # cell j has convergent input
                    mu = int(idx_post / n_cells[post] * n_cells[pre])
                    p = (n_pre / pc_ratio) / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
                    idx_pre, = np.where((np.random.uniform(size=n_cells[pre]) <= p) & place_cell[pre])
                    #print(f'Connecting {idx_pre.size} presynaptic place cells to postsynaptic neuron #{idx_post}.')
                    synapses[pre][post].connect(i=idx_pre, j=idx_post)
                else:
                    # cell j does not have convergent input
                    idx_pre, = np.where(rnd(size=n_cells[pre]) <= prob)
        else:
            if pre == post:
                synapses[pre][post].connect(condition='i!=j', p=prob)
            else:
                synapses[pre][post].connect(p=prob)
        synapses[pre][post].w = w_init

    net = Network()
    for group in spike_gen_group.values():
        net.add(group)
    for pre,post in zip(connections['pre'], connections['post']):
        net.add(synapses[pre][post])

    net.run(t_end * second, report='text')

    weights = {pre: {post: np.zeros((n_cells[pre], n_cells[post])) for post in connections['post']} for pre in connections['pre']}
    weights_sparse = {pre: {} for pre in connections['pre']}
    for pre,post in zip(connections['pre'], connections['post']):
        syn = synapses[pre][post]
        weights[pre][post][syn.i[:], syn.j[:]] = syn.w[:]
        W = weights[pre][post] * scale_factor
        np.fill_diagonal(W, 0.0)
        i,j = np.where(W > 0)
        weights_sparse[pre][post] = sparse.coo_matrix((W[i,j], (i,j)), shape=(n_cells[pre], n_cells[post]))

    np.savez_compressed(output_file, weights=weights_sparse, config=config)

