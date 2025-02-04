# -*- coding: utf-8 -*-
"""
Loads in hippocampal like spike train (produced by `generate_spike_train.py`) and runs STD learning rule in a recurrent spiking neuron population
-> creates weight matrix for PC population, used by `spw*` scripts
updated to produce symmetric STDP curve as reported in Mishra et al. 2016 - 10.1038/ncomms11552
authors: András Ecker, Eszter Vértes, last update: 11.2017
"""

import os, sys, warnings
import numpy as np
from scipy import sparse
import random as pyrandom
from brian2 import *
import matplotlib.pyplot as plt
from helper import load_spike_trains, save_wmx, make_filename
from plots import plot_STDP_rule, plot_wmx, plot_wmx_avg, plot_w_distr, save_selected_w, plot_weights

warnings.filterwarnings('ignore')
base_path = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
connection_prob_PC = 0.1


def learning(nPCs, spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init, t_max, t_step=None):
    """
    Takes a spiking group of neurons, connects the neurons sparsely with each other, and learns the weight 'pattern' via STDP:
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param nPCs: number of pyramidal cells
    :param spiking_neurons, spike_times: np.arrays for Brian2's SpikeGeneratorGroup (list of lists created by `generate_spike_train.py`) - spike train used for learning
    :param taup, taum: time constant of weight change (in ms)
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight (in S)
    :param w_init: initial weights (in S)
    :param t_max: final simulation time (in s)
    :param t_step: duration of the simulation blocks in which t_max should be subdivided (in s, if None, run a single simulation)
    :return weightmx: learned synaptic weights
    """

    if t_step is None:
        set_device('cpp_standalone')  # speed up the simulation with generated C++ code

    np.random.seed(12345)
    pyrandom.seed(12345)

    #plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, 'STDP_rule')

    PC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)

    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    STDP = Synapses(PC, PC,
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
            """)

    STDP.connect(condition='i!=j', p=connection_prob_PC)
    STDP.w = w_init

    net = Network()
    net.add(PC)
    net.add(STDP)

    weightmx_sparse = []
    got_last = False
    if t_step is None:
        net.run(t_max * second, report='text')
    else:
        n_steps = int(t_max / t_step)
        for i in range(n_steps):
            dirname = f'block_{i+1}'
            net.run(t_step * second, report='text')
            W = np.zeros((nPCs, nPCs))
            W[STDP.i[:], STDP.j[:]] = STDP.w[:]
            np.fill_diagonal(W, 0.0)
            i,j = np.where(W > 0)
            weightmx_sparse.append(sparse.coo_matrix((W[i,j], (i,j)), shape=(nPCs, nPCs)))
        if n_steps * t_step < t_max:
            net.run((t_max - n_steps * t_step) * second, report='text')
        else:
            got_last = True

    if not got_last:
        weightmx = np.zeros((nPCs, nPCs))
        weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]
        np.fill_diagonal(weightmx, 0.0)
        i,j = np.where(weightmx > 0)
        weightmx_sparse.append(sparse.coo_matrix((weightmx[i,j], (i,j)), shape=(nPCs, nPCs)))
    else:
        weightmx = W

    return weightmx, weightmx_sparse


if __name__ == '__main__':

    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = 'sym'
    assert STDP_mode in ['asym', 'sym']

    n_neurons = 8000
    place_cell_ratio = 0.5
    t_max = 405
    t_end = t_max - 5
    t_step = None
    linear = True
    f_in = make_filename('spike_trains', n_neurons, place_cell_ratio, t_max, linear, '.npz')

    # STDP parameters (see `optimization/analyse_STDP.py`)
    if STDP_mode == 'asym':
        taup = taum = 20 * ms
        Ap = 0.01
        Am = -Ap
        wmax = 4e-8  # S
        scale_factor = 1.27
    elif STDP_mode == 'sym':
        taup = taum = 62.5 * ms
        Ap = Am = 4e-3
        wmax = 2e-8  # S
        scale_factor = 0.62
    w_init = 1e-10  # S
    Ap *= wmax; Am *= wmax  # needed to reproduce Brian1 results

    npzf_name = os.path.join(base_path, 'files', f_in)
    spiking_neurons, spike_times = load_spike_trains(npzf_name)
    weightmx,weightmx_sparse = learning(n_neurons, spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init, t_end, t_step)
    weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)

    f_out = make_filename(f'wmx_{STDP_mode}', n_neurons, place_cell_ratio, t_end, linear, '.pkl')
    save_wmx(weightmx, os.path.join(base_path, 'files', f_out))

    for i,W in enumerate(weightmx_sparse):
        if t_step is not None:
            f_out = make_filename(f'wmx_{STDP_mode}', n_neurons, place_cell_ratio, np.min([(i+1) * t_step, t_end]), linear, '_sparse.pkl')
        else:
            f_out = make_filename(f'wmx_{STDP_mode}', n_neurons, place_cell_ratio, t_end, linear, '_sparse.pkl')
        save_wmx(W * scale_factor, os.path.join(base_path, 'files', f_out))

    plot_wmx(weightmx, save_name=f_out[:-4])
    plot_wmx_avg(weightmx, n_pops=100, save_name='%s_avg' % f_out[:-4])
    plot_w_distr(weightmx, save_name='%s_distr' % f_out[:-4])
    selection = np.array([500, 2400, 4000, 5500, 7015])
    plot_weights(save_selected_w(weightmx, selection), save_name='%s_sel_weights' % f_out[:-4])
    #plt.show()
