import pandas as pd
from laspy.file import File
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import gamma, norm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os

from open_las import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





path = "/home/ign.fr/ekalinicheva/DATASET_regression/"
path = "/home/ekaterina/DATASET_regression/"

gt_file = "resultats_placettes_recherche1.csv"
las_folder = path + "placettes/"

# # We open las files and create a training dataset
df_gt = pd.read_csv(path + gt_file, sep=',', header=0)  # we open GT file
all_points, _, _ = open_las(path, las_folder, gt_file)

placettes = df_gt['Name'].to_numpy()


z_all = all_points[:, 2]
z_big = z_all[z_all>=0.7]
z_small = z_all[z_all<0.7]


p = 0.95
z_med_high = z_big.copy()
z_ground = np.empty((0))
for h in range(7):
    h_int1, h_int2 = h*0.1, (h+1)*0.1
    print("interval", h_int1, h_int2)
    z_range = z_small[z_small>=h_int1]
    z_range = z_range[z_range < h_int2]
    nbr_z_range = len(z_range)
    order = np.random.permutation(np.arange(nbr_z_range))

    if h < 2:
        p1 = p - h * 0.1
    else:
        p1 = p - h * 0.1
    z_range_ground = z_range[order[:int(p1 * nbr_z_range)]]
    z_range_not_ground = z_range[order[int(p1 * nbr_z_range):]]
    z_med_high = np.concatenate((z_med_high, z_range_not_ground),0)
    z_ground = np.concatenate((z_ground, z_range_ground),0)


a_g, loc_g, scale_g = 241.3277095800471, -2.3781784017969674, 0.011351237047546814
a_v, loc_v, scale_v = 1.8275764374646228, -5.0227661734439534e-05, 3.248289124405334


params = {'phi': len(z_med_high)/(len(z_all)),
          'a_g': a_g,
          'a_v': a_v,
          'loc_g': loc_g,
          'loc_v': loc_v,
          'scale_g': scale_g,
          'scale_v': scale_v}



def e_step(x, params):
    log_p_y_x = np.log([1-params["phi"], params["phi"]])[np.newaxis, ...] + \
                np.log([gamma(params["a_g"], params["loc_g"], params["scale_g"]).pdf(x),
                        gamma(params["a_v"], params["loc_v"], params["scale_v"]).pdf(x)]).T
    log_p_y_x_norm = logsumexp(log_p_y_x, axis=1)
    return log_p_y_x_norm, np.exp(log_p_y_x - log_p_y_x_norm[..., np.newaxis])




def m_step(x, params):
    total_count = x.shape[0]
    _, heuristics = e_step(x, params)

    heuristic0 = heuristics[:, 0]
    heuristic1 = heuristics[:, 1]

    which_distr = np.zeros((len(heuristic0)))
    which_distr[heuristic1>=0.5] = 1
    which_distr[x >= 0.5] = 1


    sum_heuristic0 = np.sum(heuristic0)
    sum_heuristic1 = np.sum(heuristic1)
    phi = (sum_heuristic1/total_count)

    data_g = x[which_distr==0]
    data_v = x[which_distr==1]


    a_g, loc_g, scale_g = gamma.fit(data_g)
    a_v, loc_v, scale_v = gamma.fit(data_v)

    params = {'phi': phi,
          'a_g': a_g,
          'a_v': a_v,
          'loc_g': loc_g,
          'loc_v': loc_v,
          'scale_g': scale_g,
          'scale_v': scale_v}

    print(params)
    return params, data_g, data_v



def get_avg_log_likelihood(x, params):
    loglikelihood, _ = e_step(x, params)
    return np.mean(loglikelihood)


def run_em(x, params):
    avg_loglikelihoods = []
    count = 0
    while True:
        avg_loglikelihood = get_avg_log_likelihood(x, params)
        avg_loglikelihoods.append(avg_loglikelihood)
        if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
            break
        params, data_g, data_v = m_step(z_all, params)
        count+=1
    print(params)
    # print("\tphi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
    #            % ((params["a_g"], params["loc_g"], params["scale_g"], gamma(params["a_v"], params["loc_v"], params["scale_v"])))
    _, posterior = e_step(z_all, params)
    forecasts = np.argmax(posterior, axis=1)
    return forecasts, posterior, avg_loglikelihoods, data_g, data_v

unsupervised_forecastsforecasts, unsupervised_posterior, unsupervised_loglikelihoods, data_g, data_v = run_em(z_all, params)
print("total steps: ", len(unsupervised_loglikelihoods))
plt.plot(unsupervised_loglikelihoods)
plt.title("unsupervised log likelihoods")
plt.savefig("unsupervised.png")
plt.show()
plt.close()



d1 = gamma(params["a_g"], params["loc_g"], params["scale_g"])
d2 = gamma(params["a_v"], params["loc_v"], params["scale_v"])

mc = [1-params["phi"], params["phi"]]

z_all = np.sort(z_all)
c1 = d1.pdf(z_all) * mc[0]
c2 = d2.pdf(z_all) * mc[1]


emp_g = gamma.rvs(params["a_g"], params["loc_g"], params["scale_g"], len(z_ground))
emp_v = gamma.rvs(params["a_v"], params["loc_v"], params["scale_v"], len(z_med_high))

import matplotlib
plt.rcParams["figure.figsize"] = (25,25)
plt.rcParams["font.size"] = 25


plt.plot(z_all, c1, label='Ground', color="brown", linewidth=4)
plt.plot(z_all, c2, label='Vegetation', color="green", linewidth=4)
plt.plot(z_all, c1 + c2, '--', label='Mixture', color="yellow", linewidth=4)
plt.hist(emp_g, bins=10, density=True, histtype='stepfilled', alpha=0.2)
plt.hist(emp_v, bins=22, density=True, histtype='stepfilled', alpha=0.2)
plt.xlim(-1, 22)
plt.legend()

plt.show()


