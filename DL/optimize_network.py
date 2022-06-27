# -*- coding: utf-8 -*-
"""
Optimizes connection parameters (synaptic weights)
authors: Bence Bagi, András Ecker last update: 12.2021
"""

import os, sys, logging
import numpy as np
import pandas as pd
import bluepyopt
import multiprocessing as mp
from network import *
from sim_evaluator import *
# add 'scripts' directory to the path (to import modules)
sys.path.insert(0, os.path.join(os.path.abspath('..'), 'scripts'))
from helper import load_wmx
from plots import plot_evolution

# print info into console
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_checkpoints(pklf_name):
    """
    Loads in saved checkpoints from pickle file (used e.g. to repeat the analysis part...)
    :param pklf_name: name of the saved pickle file
    :return: obejects saved by BluePyOpt"""
    import pickle
    with open(pklf_name, "rb") as f:
        cp = pickle.load(f)
    return cp["generation"], cp["halloffame"], cp["logbook"], cp["history"]


def hof2csv(pnames, hof, f_name):
    """
    Creates pandas DaataFrame from hall of fame and saves it to csv
    :param pnames: names of optimized parameters
    :param hof: BluePyOpt HallOfFame object
    :param f_name: name of the saved file
    """
    data = np.zeros((len(hof), len(pnames)))
    for i in range(len(hof)):
        data[i, :] = hof[i]
    df = pd.DataFrame(data=data, columns=pnames)
    df.to_csv(f_name)


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
        ('rate_MF_RS', 0.1, 5),
        ('w_MF_RS', 1, 50),
        ('w_RS_BC', 1, 10),
        ('w_IB_BC', 1, 10),
        ('w_BC_RS', 1, 10),
        ('w_BC_IB', 1, 10),
        ('w_BC_BC', 1, 10),
        ('scale_RS_RS', 0.1, 5),
        ('scale_RS_IB', 0.1, 5),
        ('scale_IB_RS', 0.1, 5),
        ('scale_IB_IB', 0.1, 5),
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


    offspring_size = 5
    max_ngen = 3

    # Create multiprocessing pool for parallel evaluation of fitness function
    n_proc = np.max([offspring_size, mp.cpu_count()-1])
    pool = mp.Pool(processes=n_proc)

    # Create BluePyOpt optimization and run
    opt = bluepyopt.optimisations.DEAPOptimisation(evaluator,
                                                   offspring_size=offspring_size,
                                                   map_function=map,
                                                   eta=20,
                                                   mutpb=0.3,
                                                   cxpb=0.7)

    print(f'Started running {offspring_size*max_ngen} simulations on {n_proc} cores...')
    checkpoint_filename = 'checkpoint'
    pop, hof, log, hist = opt.run(max_ngen=max_ngen, cp_filename=checkpoint_filename)
    #del pool, opt


if __name__ == "__oldmain__":
    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["sym", "asym"]
    linear = True
    place_cell_ratio = 0.5
    f_in = "wmx_%s_%.1f_linear.pkl" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    cp_f_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", "checkpoint_%s" % f_in[4:])
    hof_f_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", "hof_%s.csv" % f_in[4:-4])

    # parameters to be fitted as a list of: (name, lower bound, upper bound)
    # the order matters! if you want to add more parameters - update `run_sim.py` too
    optconf = [("w_PC_I_", 0.1, 2.0),
               ("w_BC_E_", 0.1, 2.0),
               ("w_BC_I_", 1.0, 8.0),
               ("wmx_mult_", 0.5, 2.0),
               ("w_PC_MF_", 15.0, 25.0),
               ("rate_MF_", 5.0, 20.0)]
    pnames = [name for name, _, _ in optconf]

    offspring_size = 50
    max_ngen = 10

    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion

    # Create multiprocessing pool for parallel evaluation of fitness function
    n_proc = np.max([offspring_size, mp.cpu_count()-1])
    pool = mp.Pool(processes=n_proc)
    # Create BluePyOpt optimization and run
    evaluator = sim_evaluator.Brian2Evaluator(linear, wmx_PC_E, optconf)
    opt = bluepyopt.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, map_function=pool.map,
                                                   eta=20, mutpb=0.3, cxpb=0.7)

    print("Started running %i simulations on %i cores..." % (offspring_size*max_ngen, n_proc))
    pop, hof, log, hist = opt.run(max_ngen=max_ngen, cp_filename=cp_f_name)
    del pool, opt
    # ====================================== end of optimization ======================================

    # summary figure (about optimization)
    plot_evolution(log.select("gen"), np.array(log.select("min")), np.array(log.select("avg")),
                   np.array(log.select("std")), "fittnes_evolution")

    # save hall of fame to csv, get best individual, and rerun with best parameters to save figures
    hof2csv(pnames, hof, hof_f_name)
    best = hof[0]
    for pname, value in zip(pnames, best):
        print("%s = %.3f" % (pname, value))
    _ = evaluator.evaluate_with_lists(best, verbose=True, plots=True)
