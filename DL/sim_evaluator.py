# -*- coding: utf-8 -*-
"""
BluePyOpt evaluator for optimization
authors: Bence Bagi, András Ecker, Szabolcs Káli last update: 12.2021
"""

import os, sys, traceback, gc
# add 'scripts' directory to the path (to import modules)
sys.path.insert(0, os.path.join(os.path.abspath('..'), 'scripts'))

import numpy as np
import bluepyopt
from network import *

from helper import preprocess_monitors
from detect_oscillations import analyse_rate, ripple, gamma
from detect_replay import slice_high_activity


__all__  = ['CA3NetworkEvaluator']


class CA3NetworkEvaluator(bluepyopt.evaluators.Evaluator):
    """Evaluator class required by BluePyOpt"""

    def __init__(self,
                 params,
                 weight_matrixes,
                 N_cells,
                 cell_eqs,
                 cell_pars,
                 connectivity,
                 sim_dur=1000,
                 objectives=None,
                 linear=True):
        """
        :param Wee: weight matrix (passing Wee with cPickle to the slaves (as BluPyOpt does) is still the fastest solution)
        :param params: list of parameters to fit - every entry must be a tuple: (name, lower bound, upper bound)
        :param linear: flag to indicate if linear or circular environment is used (as oscillation detection sligthly differs)
        :param sim_dur: duration of the simulation (in ms)
        """
        super(CA3NetworkEvaluator, self).__init__()
        # Parameters to be optimized
        self.params = [bluepyopt.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in params]
        self.W            = weight_matrixes
        self.N_cells      = N_cells
        self.cell_eqs     = cell_eqs
        self.cell_pars    = cell_pars
        self.connectivity = connectivity
        self.sim_dur      = sim_dur
        self.linear       = linear
        if objectives is None or (isinstance(objectives,str) and objectives == 'default'):
            self.objectives = ['rate:RS', 'rate:IB',
                               'ripple_peak:RS', 'ripple_peak:BC',
                               'ripple_to_gamma_ratio:RS', 'ripple_to_gamma_ratio:BC',
                               'no_gamma_peak:BC']
        else:
            self.objectives = objectives
        self.n_objectives = len(self.objectives)
        self.cell_types = self.N_cells.keys()

        self.target_rates = {'RS': 3, 'IB': 2}

    def generate_model(self, individual, verbose=False):
        """Runs single simulation (see `network.py`) and returns monitors"""        
        poisson_rates = {'MF': {'RS': individual[0]}}
        syn_weights = {
            'MF': {'RS': individual[1]},
            'RS': {'BC': individual[2]},
            'IB': {'BC': individual[3]},
            'BC': {'RS': individual[4],
                   'IB': individual[5],
                   'BC': individual[6]}
        }
        scaling_factors = {
            'RS': {'RS': individual[7],
                   'IB': individual[8]},
            'IB': {'RS': individual[9],
                   'IB': individual[10]}
        }
        return run_net_sim(self.W,
                           scaling_factors,
                           syn_weights,
                           poisson_rates,
                           self.N_cells,
                           self.cell_eqs,
                           self.cell_pars,
                           self.connectivity,
                           tend=self.sim_dur,
                           rnd_seed=123456,
                           record_state=False,
                           full_output=False,
                           verbose=verbose)


    def evaluate_with_lists(self, individual, verbose=False, plots=False):
        """Fitness error used by BluePyOpt for the optimization"""
        spike_monitors, rate_monitors = self.generate_model(individual, verbose)

        if plots:
            from brian2tools import plot_raster, plot_rate
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(2, 1)
            cell_type = 'BC'
            plot_raster(spike_monitors[cell_type].i, spike_monitors[cell_type].t, color='k', markersize=1, axes=ax[0])
            plot_rate(rate_monitors[cell_type].t, rate_monitors[cell_type].smooth_rate(window='flat', width=10.1*ms),
                      linewidth=1, color='k', axes=ax[1])
            plt.show()

        
        try:
            num_spikes = np.array(list(map(lambda spks: spks.num_spikes, spike_monitors.values())))
            if all(num_spikes > 0):  # check if there is any activity
                # analyse spikes
                spike_times, spiking_neurons, rates = {}, {}, {}
                ISI_hist, bin_edges = {}, {}
                slice_idx = {}
                mean_rate, rate_ac, max_ac, t_max_ac = {}, {}, {}, {},
                freq, Pxx = {}, {}
                avg_ripple_freq, ripple_power = {}, {}
                avg_gamma_freq, gamma_power = {}, {}
                for cell_type in self.cell_types:
                    # analyse spikes
                    spike_times[cell_type], \
                        spiking_neurons[cell_type], \
                        rates[cell_type], \
                        ISI_hist[cell_type], \
                        bin_edges[cell_type] = preprocess_monitors(spike_monitors[cell_type],
                                                                   rate_monitors[cell_type],
                                                                   calc_ISI=True)
                    slice_idx[cell_type] = [] if not self.linear else slice_high_activity(rates[cell_type],
                                                                                          th=2,
                                                                                          min_len=260,
                                                                                          sim_dur=self.sim_dur)
                    # analyse rates
                    mean_rate[cell_type], \
                        rate_ac[cell_type], \
                        max_ac[cell_type], \
                        t_max_ac[cell_type], \
                        freq[cell_type], \
                        Pxx[cell_type] = analyse_rate(rates[cell_type], 1000.0, slice_idx[cell_type])

                    avg_ripple_freq[cell_type], ripple_power[cell_type] = ripple(freq[cell_type],
                                                                                 Pxx[cell_type],
                                                                                 slice_idx[cell_type])
                    avg_gamma_freq[cell_type], gamma_power[cell_type] = gamma(freq[cell_type],
                                                                              Pxx[cell_type],
                                                                              slice_idx[cell_type])
                del spike_monitors, rate_monitors
                gc.collect()

                errors = []
                for obj in self.objectives:
                    obj_name, cell_type = obj.split(':')
                    if obj_name == 'rate' and cell_type in self.target_rates:
                        # look for 'low' excitatory population rates (as specified in self.target_rates)
                        rate_error = np.exp(-0.5 * (mean_rate[cell_type] - self.target_rates[cell_type]) ** 2)
                        errors.append(rate_error)
                        if verbose:
                            print(f'Mean rate for cell type `{cell_type}`: {mean_rate[cell_type]:.2f} spike/s ' + \
                                  f'(target: {self.target_rates[cell_type]:.2f}, error: {rate_error:.2e}).')
                    elif obj_name == 'ripple_peak':
                        # look for significant ripple peak close to 180 Hz
                        ripple_peak = np.exp(-0.5 * (avg_ripple_freq[cell_type] - 180.) ** 2 / 20 ** 2) \
                                      if not np.isnan(avg_ripple_freq[cell_type]) else 0.
                        if cell_type == 'BC':
                            ripple_peak *= 2
                        errors.append(ripple_peak)
                        if verbose:
                            print(f'Average ripple frequency for cell type `{cell_type}`: {avg_ripple_freq[cell_type]:.3f} Hz.')
                            print(f'Ripple power for cell_type `{cell_type}`: {ripple_power[cell_type]:.3f}.')
                    elif obj_name ==  'ripple_to_gamma_ratio':
                        # look for high ripple/gamma power ratio
                        if cell_type in ('RS','IB'):
                            ripple_ratio = np.clip(ripple_power[cell_type] / gamma_power[cell_type], 0., 5.)
                        else:
                            ripple_ratio = np.clip(2 * ripple_power[cell_type] / gamma_power[cell_type], 0., 10.)
                        errors.append(ripple_ratio)
                        if verbose:
                            print(f'Average ripple to gamma ratio for cell type `{cell_type}`: {ripple_ratio:.3f}.')
                    elif obj_name  == 'no_gamma_peak':
                        # penalize gamma peak (in inhibitory pop) - binary variable, which might not be the best for this algo.
                        gamma_peak = 1. if np.isnan(avg_gamma_freq[cell_type]) else 0.
                        errors.append(gamma_peak)
                        if verbose:
                            if gamma_peak == 1.:
                                print(f'Cell type `{cell_type}` has a significant gamma peak.')
                            else:
                                print(f'Cell type `{cell_type}` does not have a significant gamma peak.')
                    else:
                        raise Exception(f'Unknown objective `{obj_name}`')

                #if plots:
                #    from plots import plot_raster, plot_PSD, plot_zoomed
                #    plot_raster(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist_PC, bin_edges_PC],
                #                slice_idx, "blue", multiplier_=1)
                #    plot_PSD(rate_PC, rate_ac_PC, f_PC, Pxx_PC, "PC_population", "blue", multiplier_=1)
                #    plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=1)
                #    _ = plot_zoomed(spike_times_PC, spiking_neurons_PC, rate_PC, "PC_population", "blue", multiplier_=1)
                #    plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green",
                #                multiplier_=1, PC_pop=False)

                # *-1 since the algorithm tries to minimize...
                return [err * -1 for err in errors]

            return [0. for _ in range(self.n_objectives)]  # worst case scenario

        except Exception:
            # Make sure exception and backtrace are thrown back to parent process
            raise Exception(''.join(traceback.format_exception(*sys.exc_info())))



if __name__ == '__main__':
    import json
    import re

    try:
        sim_dur = float(sys.argv[1]) * 1000
    except:
        sim_dur = 500

    config = json.load(open('configs/sim_pars.json'))
    config_files = {re.findall('[a-zA-Z]+', k)[0]: v for k,v in config.items() if 'config_file' in k}
    cell_types = list(config_files.keys())
    cell_pars = {cell_type: json.load(open(f, 'r')) for cell_type, f in config_files.items()}
    N_cells = {cell_type: config[f'N_{cell_type}'] for cell_type in cell_types}
    cell_eqs = {cell_type: AdEx_eqs_with_MF.format(cell_type, *cell_types) if cell_type == 'RS' \
                else AdEx_eqs_without_MF.format(cell_type, *cell_types) for cell_type in cell_types}

    params = [
        ['rate_MF_RS', 0.1, 5],
        ['w_MF_RS', 1, 50],
        ['w_RS_BC', 1, 10],
        ['w_IB_BC', 1, 10],
        ['w_BC_RS', 1, 10],
        ['w_BC_IB', 1, 10],
        ['w_BC_BC', 1, 10],
        ['scale_RS_RS', 0.1, 5],
        ['scale_RS_IB', 0.1, 5],
        ['scale_IB_RS', 0.1, 5],
        ['scale_IB_IB', 0.1, 5],
    ]

    data = np.load(config['weights'], allow_pickle=True)
    weight_matrixes = data['weights'].item()

    objectives = ['rate:RS', 'rate:IB']
    evaluator = CA3NetworkEvaluator(params,
                                    weight_matrixes,
                                    N_cells,
                                    cell_eqs,
                                    cell_pars,
                                    config['connectivity'],
                                    sim_dur=sim_dur,
                                    objectives=objectives,
                                    linear=True)

    individual = [
        0.5, # rate MF -> RS
        40,  # weight MF -> RS
        10,  # weight RS -> BC
         1,  # weight IB -> BC
        10,  # weight BC -> RS
         1,  # weight BC -> IB
         1,  # weight BC -> BC
         5,  # scaling factor RS -> RS
        25,  # scaling factor RS -> IB
         1,  # scaling factor IB -> RS
         1,  # scaling factor IB -> IB
    ]

    err = evaluator.evaluate_with_lists(individual, verbose=True, plots=False)

