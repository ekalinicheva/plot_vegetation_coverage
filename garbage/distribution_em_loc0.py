import pandas as pd
from laspy.file import File
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import seaborn as sns
from useful_functions import *
from scipy.stats import gamma, norm
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




# Print stats to file
def print_stats(stats_file, text, print_to_console=True):
    with open(stats_file, 'a') as f:
        if isinstance(text, list):
            for t in text:
                f.write(t + "\n")
                if print_to_console:
                    print(t)
        else:
            f.write(text + "\n")
            if print_to_console:
                print(text)
    f.close()

#Function to create a new folder if not exists
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
# %%




path = "/home/ign.fr/ekalinicheva/DATASET_regression/"
# path = "/home/ekaterina/DATASET_regression/"

gt_file = "resultats_placettes_recherche1.csv"
las_folder = path + "placettes/"



df_gt = pd.read_csv(path + gt_file, sep=',', header=0)

# xyzi = np.empty((0, 5))
dataset = {}
las_files = os.listdir(las_folder)
all_points = np.empty((0, 9))
for las_file in las_files:
    las = File(las_folder + las_file, mode='r')
    # print(las_file)
    las = File(las_folder + las_file, mode='r')
    x_las = las.X
    y_las = las.Y
    z_las = las.Z
    r = las.Red
    g = las.Green
    b = las.Blue
    nir = las.nir
    intensity = las.intensity
    nbr_returns = las.num_returns
    # print(intensity.max())
    # cl = laz.classification
    points_placette = np.asarray([x_las / 100, y_las / 100, z_las / 100, r, g, b, nir, intensity, nbr_returns]).T
    # There is a file with 2 points 60m above others (maybe birds), we delete these points
    if las_file == "Releve_Lidar_F70.las":
        points_placette = points_placette[points_placette[:, 2] < 640]
    # We do the same for the intensity
    if las_file == "Releve_Lidar_F39.las":
        points_placette = points_placette[points_placette[:, -2] < 20000]
    # all_points = np.append(all_points, points_placette, axis=0)
    # dataset[os.path.splitext(las_file)[0]] = points_placette

    # We directly substract z_min at local level
    xyz = points_placette[:, :3]
    knn = NearestNeighbors(100, algorithm='kd_tree').fit(xyz[:, :2])
    _, neigh = knn.radius_neighbors(xyz[:, :2], 0.75)
    z = xyz[:, 2]
    zmin_neigh = []
    for n in range(len(z)):
        zmin_neigh.append(np.min(z[neigh[n]]))

    points_placette[:, 2] = points_placette[:, 2] - zmin_neigh
    all_points = np.append(all_points, points_placette, axis=0)
    dataset[os.path.splitext(las_file)[0]] = points_placette


# %%
placettes = df_gt['Name'].to_numpy()
# order = np.random.permutation(np.arange(len(placettes)))
#
# train_list = placettes[order[:int(0.7 * len(placettes))]]
# test_list = placettes[order[int(0.7 * len(placettes)):]]

dataset_norm = {}
gt_all = np.zeros((len(placettes), 3), dtype=float)
abs_error_all = []
predictions = np.zeros((len(placettes), 3), dtype=float)
for p in range(len(placettes)):
    pl = placettes[p]
    cloud_data = dataset[pl].copy()

    xmean, ymean = np.asarray(cloud_data[:, :2]).mean(0)
    zmin = cloud_data[:, 2].min()
    #normalizing data
    cloud_data[:, 0] = (cloud_data[:, 0] - xmean)/10 #x
    cloud_data[:, 1] = (cloud_data[:, 1] - ymean)/10 #y
    # cloud_data[2] = (cloud_data[2] - zmin)/22 #z
    # cloud_data[:, 2] = (cloud_data[:, 2] - zmin) #z
    z = cloud_data[:, 2]
    # print(np.sort(cloud_data[:, 2]))

    colors_max = 65536
    cloud_data[:, 3:7] = cloud_data[:, 3:7]/colors_max
    int_max = 32768
    cloud_data[:, 7] = cloud_data[:, 7] / int_max
    cloud_data[:, 8] = (cloud_data[:, 8] - 1)/(7-1)

    dataset_norm[pl] = cloud_data

    # sns.displot(z[z>0.5])
    # plt.show()



all_points_norm = np.empty((0, 9))
for key in dataset_norm.keys():
    points_norm = dataset_norm.get(key)
    all_points_norm = np.append(all_points_norm, points_norm, axis=0)




z_all = all_points_norm[:, 2]



z_big = z_all[z_all>=0.7]
z_small = z_all[z_all<0.7]
# z_smaller025 = z_small[z_small<=0.25]
# z_bigger025 = z_small[z_small>0.25]
p=0.95
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
        p1 = p-h*0.1
    else:
        p1 = p - h * 0.1
    z_range_ground = z_range[order[:int(p1 * nbr_z_range)]]
    z_range_not_ground = z_range[order[int(p1 * nbr_z_range):]]
    # print(len(z_range_ground), len(z_range_not_ground))
    z_med_high = np.concatenate((z_med_high, z_range_not_ground),0)
    z_ground = np.concatenate((z_ground, z_range_ground),0)



z_all = np.asarray([0, 0.1, 0.2, 0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,3, 4, 5])

a_g, loc_g, scale_g = 241.3277095800471, -2.3781784017969674, 0.011351237047546814
a_v, loc_v, scale_v = 1.8275764374646228, -5.0227661734439534e-05, 3.248289124405334


params = {'phi': len(z_med_high)/(len(z_all)),
          'a_g': a_g,
          'a_v': a_v,
          'loc_g': 0,
          'loc_v': 0,
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

    phi = len(data_v)/(len(data_v)+len(data_g))
    print(data_g)
    print(data_v)


    a_g, loc_g, scale_g = gamma.fit(data_g, loc=0)
    a_v, loc_v, scale_v = gamma.fit(data_v, loc=0)
    print(loc_g)
    print(loc_v)

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
    while count<4:
        avg_loglikelihood = get_avg_log_likelihood(x, params)
        avg_loglikelihoods.append(avg_loglikelihood)
        print(avg_loglikelihood)
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

#
# with open('zzz.npy', 'wb') as f:
#     np.save(f, z_all)
#
# exit()

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


