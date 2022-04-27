
import numpy as np

__all__ = ['make_filename', 'rasterplot', 'psth',
           'make_gamma_spike_train', 'make_inhomogeneous_gamma_spike_train',
           'make_poisson_spike_train', 'make_inhomogeneous_poisson_spike_train']

def make_filename(prefix, n_neurons, place_cell_ratio, dur, linear, suffix):
    fname = f"{prefix}_N={n_neurons}_ratio={place_cell_ratio:.1f}_dur={dur:.0f}"
    if linear:
        fname += "_linear"
    return fname + suffix


def rasterplot(spike_trains, max_neurons=None, ax=None, color='k', marker='.', markersize=2):
    if max_neurons is None or max_neurons > len(spike_trains):
        max_neurons = len(spike_trains)
    if ax is None:
        ax = plt.gca()
    n_spikes = list(map(len, spike_trains))
    for i in range(max_neurons):
        ax.plot(spike_trains[i], i + 1 + np.zeros(n_spikes[i]), marker, color=color, markersize=markersize)


def psth(spike_trains, binwidth, interval=None):
    if interval is None:
        first = np.min(list(map(lambda x: x[0], spike_trains)))
        last = np.max(list(map(lambda x: x[-1], spike_trains)))
    else:
        first, last = interval
    edges = np.r_[first : last : binwidth]
    n_edges = edges.size
    n_trials = len(spike_trains)
    count = np.zeros((n_trials, n_edges - 1))
    for i in range(n_trials):
        count[i,:],_ = np.histogram(spike_trains[i], edges)
    nu = count.sum(axis=0) / (n_trials * binwidth)
    return nu, edges[:-1], count


def make_gamma_spike_train(k, rate, tend=None, Nev=None, refractory_period=0, random_state=None):
    from scipy.stats import gamma
    if Nev is None:
        Nev = int(np.ceil(tend * rate))
    ISIs = gamma.rvs(k, loc=0, scale=1 / (k * rate), size=int(1.2 * Nev), random_state=random_state)
    ISIs = ISIs[ISIs > refractory_period]
    spks = np.cumsum(ISIs)
    if tend is not None:
        spks = spks[spks <= tend]
    else:
        spks = spks[:Nev]
    return spks


def make_inhomogeneous_gamma_spike_train(k, rate_fun, max_rate, tend=None, Nev=None,
                                         refractory_period=0, random_state=None, full_output=False):
    # first generate homogeneous gamma spike times with no refractory period: we will enforce it later
    all_spikes = make_gamma_spike_train(k, max_rate, tend, Nev, refractory_period=0, random_state=random_state)
    prob = rate_fun(all_spikes) / max_rate
    if random_state == None:
        rnd = np.random.uniform(size=prob.shape)
    else:
        rnd = random_state.uniform(size=prob.shape)
    spikes = all_spikes[rnd < prob]
    if len(spikes) > 0:
        ISIs = np.diff(spikes)
        spikes = spikes[0] + np.concatenate([[0], np.cumsum(ISIs[ISIs > refractory_period])])
    if not full_output:
        return spikes
    return spikes, all_spikes


make_poisson_spike_train = lambda rate, tend=None, Nev=None, refractory_period=0, random_state=None: \
    make_gamma_spike_train(1, rate, tend, Nev, refractory_period, random_state)


make_inhomogeneous_poisson_spike_train = lambda rate_fun, max_rate, tend=None, Nev=None, \
    refractory_period=0, random_state=None, full_output=False: \
    make_inhomogeneous_gamma_spike_train(1, rate_fun, max_rate, tend, Nev, \
                                         refractory_period, random_state, full_output)
