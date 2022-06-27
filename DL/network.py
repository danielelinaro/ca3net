# -*- coding: utf-8 -*-

import os
from itertools import chain
import numpy as np
import random as pyrandom
from brian2 import *
import platform
if platform.system() == 'Linux':
    set_device('cpp_standalone')
else:
    prefs.codegen.target = 'numpy'  #cython  # weave is not multiprocess-safe!


__all__ = ['AdEx_eqs_with_MF', 'AdEx_eqs_without_MF', 'AdEx_vars_units', 'run_net_sim']



AdEx_eqs_with_MF = """
dvm/dt = (-gL_{0}*(vm-EL_{0}) + gL_{0}*Delta_T_{0}*exp((vm-V_th_{0})/Delta_T_{0}) - w - ((g_ampa_{1}+g_ampa_{2}+g_ampa_MF)*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_{0} : volt (unless refractory)
dw/dt = (a_{0}*(vm-EL_{0}) - w) / tau_w_{0} : amp
dg_ampa_{1}/dt = (x_ampa_{1} - g_ampa_{1}) / rise_{1}_{0} : 1
dx_ampa_{1}/dt = -x_ampa_{1} / decay_{1}_{0} : 1
dg_ampa_{2}/dt = (x_ampa_{2} - g_ampa_{2}) / rise_{2}_{0} : 1
dx_ampa_{2}/dt = -x_ampa_{2} / decay_{2}_{0} : 1
dg_ampa_MF/dt = (x_ampa_MF - g_ampa_MF) / rise_MF_{0} : 1
dx_ampa_MF/dt = -x_ampa_MF / decay_MF_{0} : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_{3}_{0} : 1
dx_gaba/dt = -x_gaba/decay_{3}_{0} : 1
"""



AdEx_eqs_without_MF = """
dvm/dt = (-gL_{0}*(vm-EL_{0}) + gL_{0}*Delta_T_{0}*exp((vm-V_th_{0})/Delta_T_{0}) - w - ((g_ampa_{1}+g_ampa_{2})*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_{0} : volt (unless refractory)
dw/dt = (a_{0}*(vm-EL_{0}) - w) / tau_w_{0} : amp
dg_ampa_{1}/dt = (x_ampa_{1} - g_ampa_{1}) / rise_{1}_{0} : 1
dx_ampa_{1}/dt = -x_ampa_{1} / decay_{1}_{0} : 1
dg_ampa_{2}/dt = (x_ampa_{2} - g_ampa_{2}) / rise_{2}_{0} : 1
dx_ampa_{2}/dt = -x_ampa_{2} / decay_{2}_{0} : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_{3}_{0} : 1
dx_gaba/dt = -x_gaba/decay_{3}_{0} : 1
"""



AdEx_vars_units = {
    'Cm': pF,
    'gL': nS,
    'EL': mV,
    'V_th': mV,
    'V_peak': mV,
    'V_reset': mV,
    'Delta_T': mV,
    'tau_w': ms,
    'a': nS,
    'b': pA,
    'tau_arp': ms,
    'Ie': pA
}
    


def run_net_sim(weight_matrixes, scaling_factors, syn_weights, poisson_rates,
                N_cells, cell_eqs, cell_pars, connectivity, tend=100,
                Erev_E = 0, Erev_I=-70, z=1, record_state=False,
                full_output=False, rnd_seed=None, verbose=False):

    tend *= ms
    Erev_E *= mV
    Erev_I *= mV
    z *= nS
    seed(rnd_seed)

    cell_types = list(N_cells.keys())
    tau_arp = {}
    for cell_type in cell_types:
        for par,value in cell_pars[cell_type].items():
            cmd = f'{par}_{cell_type} = {value} * AdEx_vars_units[par]'
            exec(cmd)
            if par == 'tau_arp':
                tau_arp[cell_type] = value * AdEx_vars_units[par]
    neuron_groups = {cell_type: NeuronGroup(N,
                                            model=cell_eqs[cell_type],
                                            threshold=f'vm>V_peak_{cell_type}',
                                            reset=f'vm=V_reset_{cell_type}; w+=b_{cell_type}',
                                            refractory=tau_arp[cell_type],
                                            method='exponential_euler',
                                            name=f'{cell_type}_neurons')
                     for cell_type, N in N_cells.items()}
    ampa_conductances = set(re.findall('g_ampa_[a-zA-Z]*', cell_eqs['RS']))
    gaba_conductances = set(re.findall('g_gaba_[a-zA-Z]*', cell_eqs['RS']))
    for cell_type in cell_types:
        neuron_groups[cell_type].vm = eval(f'EL_{cell_type}')
        for cond in chain(ampa_conductances, gaba_conductances):
            try:
                setattr(neuron_groups[cell_type], cond, 0.)
            except:
                if verbose: print(f'Neuron group `{cell_type}` does not have a `{cond}` conductance')

    poisson_groups = {pre: {} for pre in connectivity['pre']}
    poisson_spike_monitors = {}
    poisson_rate_monitors = {}
    synapses = {pre: {} for pre in connectivity['pre']}
    for pre,post,prob,delay,rise,decay,trans in zip(connectivity['pre'], connectivity['post'],
                                                    connectivity['prob'], connectivity['delay'],
                                                    connectivity['tau_rise'], connectivity['tau_decay'],
                                                    connectivity['transmitter']):
        tp = (decay * rise) / (decay - rise) * np.log(decay / rise)
        norm_coeff = 1.0 / (np.exp(-tp/decay) - np.exp(-tp/rise))
        exec(f'norm_{pre}_{post} = norm_coeff')
        exec(f'decay_{pre}_{post} = decay * ms')
        exec(f'rise_{pre}_{post} = rise * ms')

        if prob is None:
            if verbose: print(f'Connections from `{pre}` to `{post}` are defined in a connection matrix')
            if trans in ('ampa', 'nmda'):
                effect = 'exc'
            elif trans == 'gaba':
                effect = 'inh'
            else:
                raise Exception(f'Unknown transmitter `{trans}`')
            update_eq = f'x_{trans}_{pre} += norm_{pre}_{post} * w_{effect}'
            synapses[pre][post] = Synapses(neuron_groups[pre],
                                           neuron_groups[post],
                                           f'w_{effect}:1',
                                           on_pre=update_eq,
                                           delay=delay*ms,
                                           name=f'syn_{pre}_{post}')
            try:
                coeff = scaling_factors[pre][post]
            except:
                coeff = 1
            W = weight_matrixes[pre][post].todense() * coeff * 1e9
            I,J = np.nonzero(W)
            synapses[pre][post].connect(i=I, j=J)
            setattr(synapses[pre][post], f'w_{effect}', W[I,J].flatten())
        else:
            exec(f'w_{pre}_{post} = syn_weights[pre][post]')
            if prob > 0:
                if trans == 'ampa':
                    update_eq = f'x_{trans}_{pre} += norm_{pre}_{post} * w_{pre}_{post}'
                else:
                    update_eq = f'x_{trans} += norm_{pre}_{post} * w_{pre}_{post}'
                synapses[pre][post] = Synapses(neuron_groups[pre],
                                               neuron_groups[post],
                                               on_pre=update_eq,
                                               delay=delay*ms,
                                               name=f'syn_{pre}_{post}')
                synapses[pre][post].connect(p=prob)
            elif prob == -1:
                poisson_groups[pre][post] = PoissonGroup(N_cells[post],
                                                         poisson_rates[pre][post] * Hz,
                                                         name=f'poisson_{pre}_{post}')
                if pre not in poisson_spike_monitors:
                    poisson_spike_monitors[pre] = {}
                    poisson_rate_monitors[pre] = {}
                poisson_spike_monitors[pre][post] = SpikeMonitor(poisson_groups[pre][post])
                poisson_rate_monitors[pre][post] = PopulationRateMonitor(poisson_groups[pre][post])
                update_eq = f'x_{trans}_{pre} += norm_{pre}_{post} * w_{pre}_{post}'
                synapses[pre][post] = Synapses(poisson_groups[pre][post],
                                               neuron_groups[post],
                                               on_pre=update_eq,
                                               name=f'syn_{pre}_{post}')
                synapses[pre][post].connect(j='i')
            else:
                raise Exception(f'Do not know what to do with prob = {prob}')

    del W

    spike_monitors = {cell_type: SpikeMonitor(group, name=f'spike_mon_{cell_type}')
                      for cell_type,group in neuron_groups.items()}
    pop_rate_monitors = {cell_type: PopulationRateMonitor(group, name=f'rate_mon_{cell_type}')
                         for cell_type,group in neuron_groups.items()}
    if record_state:
        state_monitors = {cell_type: StateMonitor(group, ['vm'], record=range(5), name=f'state_{cell_type}')
                          for cell_type,group in neuron_groups.items()}
    net = Network()
    for group in neuron_groups.values():
        net.add(group)
    for pre,groups in poisson_groups.items():
        for post,group in groups.items():
            if verbose: print(f'Adding Poisson group from `{pre}` to `{post}` to the network')
            net.add(group)
            net.add(poisson_spike_monitors[pre][post])
            net.add(poisson_rate_monitors[pre][post])
    for pre,groups in synapses.items():
        for post,group in groups.items():
            if verbose: print(f'Adding synapses from `{pre}` to `{post}` to the network')
            net.add(group)
    for cell_type in cell_types:
        if verbose: print(f'Adding monitors to the network')
        net.add(spike_monitors[cell_type])
        net.add(pop_rate_monitors[cell_type])
        if record_state:
            net.add(state_monitors[cell_type])

    if verbose:
        net.run(tend, report='text')
    else:
        net.run(tend)

    res = spike_monitors, pop_rate_monitors
    if record_state:
        res += state_monitors,
    if full_output:
        res += poisson_spike_monitors, poisson_rate_monitors
    return res



if __name__ == '__main__':

    import json
    import re
    from brian2tools import plot_raster, plot_rate
    config = json.load(open('configs/sim_pars.json'))
    config_files = {re.findall('[a-zA-Z]+', k)[0]: v for k,v in config.items() if 'config_file' in k}
    cell_types = list(config_files.keys())
    cell_pars = {cell_type: json.load(open(f, 'r')) for cell_type, f in config_files.items()}
    N_cells = {cell_type: config[f'N_{cell_type}'] for cell_type in cell_types}
    cell_eqs = {cell_type: AdEx_eqs_with_MF.format(cell_type, *cell_types) if cell_type == 'RS' \
                else AdEx_eqs_without_MF.format(cell_type, *cell_types) for cell_type in cell_types}

    poisson_rates = {'MF': {'RS': 0.5}}
    syn_weights = {
        'MF': {'RS': 40},
        'RS': {'BC': 10},
        'IB': {'BC': 1},
        'BC': {'RS': 10, 'IB': 1, 'BC': 1}
    }

    data = np.load(config['weights'], allow_pickle=True)
    weight_matrixes = data['weights'].item()
    scaling_factors = {pre: {post: 1 for post in weight_matrixes[pre]} for pre in weight_matrixes}
    scaling_factors['RS']['RS'] = 5
    scaling_factors['RS']['IB'] = 25
    spikes,rates,Vm,poisson_spikes,poisson_rates = run_net_sim(weight_matrixes,
                                                               scaling_factors,
                                                               syn_weights,
                                                               poisson_rates,
                                                               N_cells,
                                                               cell_eqs,
                                                               cell_pars,
                                                               config['connectivity'],
                                                               tend=250,
                                                               record_state=True,
                                                               full_output=True,
                                                               rnd_seed=123456,
                                                               verbose=True)

    r,c = 3,4
    w,h = 3,2
    fig,ax = plt.subplots(r, c, figsize=(c*w, r*h), sharex=True)

    cmap = 'krbg'
    for i,cell_type in enumerate(cell_types):
        ax[0,i+1].set_title(cell_type)
        ax[0,i+1].plot(Vm[cell_type].t / ms, Vm[cell_type].vm.T / mV, cmap[i], lw=1)
        plot_raster(spikes[cell_type].i, spikes[cell_type].t, color=cmap[i],
                    axes=ax[1,i+1], markersize=1)
        plot_rate(rates[cell_type].t, rates[cell_type].smooth_rate(window='flat',
                                                                   width=10.1*ms),
                  linewidth=1, color=cmap[i], axes=ax[2,i+1])
    ax[1,0].set_title('MF')
    mon = poisson_spikes['MF']['RS']
    plot_raster(mon.i, mon.t, color=cmap[-1], axes=ax[1,0], markersize=1)
    mon = poisson_rates['MF']['RS']
    plot_rate(mon.t, mon.smooth_rate(window='flat', width=10.1*ms),
              linewidth=1, color=cmap[-1], axes=ax[2,0])

    for i in range(r):
        for j in range(c):
            for side in 'right','top':
                ax[i,j].spines[side].set_visible(False)
            if i != r-1:
                ax[i,j].set_xlabel('')
            if i == 0 and j > 0:
                ax[i,j].set_ylabel('Vm (mV)')
            if i == 1 and j > 0:
                ax[i,j].set_ylim([0, N_cells[cell_types[j-1]]])
    ax[1,0].set_ylim([0, N_cells['RS']])
    ax[0,0].set_visible(False)
    fig.tight_layout()
    fig.savefig('network_sim.pdf')

