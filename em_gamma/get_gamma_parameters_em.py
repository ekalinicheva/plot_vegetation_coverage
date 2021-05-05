import argparse
import numpy as np
from scipy.stats import gamma
from scipy.special import digamma, polygamma
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
def get_gamma_parameters(all_z):
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    # Optimization arguments
    parser.add_argument('--ECM_ite_max', default=5, type=int, help='Max number of EVM iteration')
    parser.add_argument('--NR_ite_max', default=10, type=int, help='Max number of Netwon-Rachson iteration')
    args = parser.parse_args()
    # #data loading
    # all_z = np.load('./zzz.npy') + 1e-2
    all_z = all_z + 1e-2

    cannot_be_low = all_z>0.5
    #initialization
    shape = np.array([0.2, 1.8])  # scale gamma parameters
    scale = np.array([0.3, 4]) #shape gamma parameters
    pi = np.array([0.5, 5]) #bernoulli parameter
    def view_distribution():
        fig, ax = plt.subplots(1, 1)
        x = np.linspace(0,10, 100)
        plt.hist(all_z, bins=100, range=(0, 10), density=True)
        plt.plot(x, pi[0] * gamma.pdf(x, shape[0], 0, scale[0]), 'r-', lw=1, label='gamma1')
        plt.plot(x, pi[1] * gamma.pdf(x, shape[1], 0, scale[1]), 'k-', lw=1, label='gamma2')
        plt.tight_layout()
        axes = plt.gca()
        axes.set_ylim([0,0.5])
        plt.show(block=True)
    def E_step():
        expected_values = np.vstack((gamma.pdf(all_z, shape[0], 0, scale[0]),gamma.pdf(all_z, shape[1], 0, scale[1])))
        expected_values = expected_values * pi[:, None]
        expected_values[0,cannot_be_low] = 0
        return expected_values/expected_values.sum(0)
    def inner_optim(expected_values):
        #find simultaneously the CM values for shape defined as poles obj, with with newton-raphson
        x = shape
        def obj(x):
            #the function to minimize
            return (expected_values * (np.log(all_z[None,:]) - np.log(scale)[:,None] - digamma(x)[:,None])).mean(1)
        def derivative(x):
            #its derivative
            return (expected_values * (- polygamma(1, x)[:, None])).mean(1)
        for sub_ite in range(args.NR_ite_max):
            print("    NR it %d - obj = %3.3f %3.3f" % (sub_ite, *obj(x)))
            if (np.abs(obj(x))<1e-3).all():
                print("Newton Rachson terminated")
                break
            x = x - obj(x) / derivative(x) #one NR iteration
        return x
    def CM1(expected_values):
        #first CM-step
        pi = expected_values.mean(1)
        scale = inner_optim(expected_values)
        return pi, scale
    def CM2(expected_values):
        #second CM-step
        num = (all_z[None,:] * expected_values).mean(1)
        denom = scale * expected_values.mean(1)
        return num/denom
    def log_likelihood():
        expected_values = np.vstack((gamma.pdf(all_z, shape[0], 0, scale[0]), gamma.pdf(all_z, shape[1], 0, scale[1])))
        expected_values = expected_values * pi[:, None]
        expected_values[0, cannot_be_low] = 0
        return -np.log(expected_values.sum(0)).mean()
    #---main loop---
    print("Likelihood at init: %2.3f" % (log_likelihood()))
    for ite in range(args.ECM_ite_max):
        expected_values = E_step()
        pi, scale = CM1(expected_values)
        shape = CM2(expected_values)
        print("Likelihood at ite %d: %2.3f" % (ite, log_likelihood()))
    # print(shape)
    # print(scale)
    # print(pi)
    # x = np.linspace(0, 10, 101)[1:]
    # histo = plt.hist(all_z, bins=100, range=(0, 10), density=True)
    # bins = histo[1][1:]
    # pdf =  histo[0]
    # y1 = pi[0] * gamma.pdf(x, shape[0], 0, scale[0])
    # y2 = pi[1] * gamma.pdf(x, shape[1], 0, scale[1])
    # np.savetxt("ECM.csv",  np.vstack((bins, pdf, y1, y2)).transpose(), delimiter=",")
    # view_distribution()
    params = {'phi': pi[0], 'a_g': shape[0], 'a_v': shape[1],
              'loc_g': 0, 'loc_v': 0, 'scale_g': scale[0],
              'scale_v': scale[1]}
    return params