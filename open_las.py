import os
import numpy as np
from laspy.file import File
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.simplefilter(action='ignore')



def open_las(path, las_folder, gt_file):
    # We open las files and create a training dataset
    dataset = {}  # dict to store numpy array with each plot separately
    mean_dataset = {}  # we keep track of plots means to reverse the normalisation in the future

    # We iterate through las files and transform them to np array
    las_files = os.listdir(las_folder)
    all_points = np.empty((0, 9))
    for las_file in las_files:
        las = File(las_folder + las_file, mode='r')
        x_las = las.X
        y_las = las.Y
        z_las = las.Z
        r = las.Red
        g = las.Green
        b = las.Blue
        nir = las.nir
        intensity = las.intensity
        nbr_returns = las.return_num
        points_placette = np.asarray([x_las / 100, y_las / 100, z_las / 100, r, g, b, nir, intensity,
                                      nbr_returns]).T  # we divide by 100 as all the values in las are in cm

        # There is a file with 2 points 60m above others (maybe birds), we delete these points
        if las_file == "Releve_Lidar_F70.las":
            points_placette = points_placette[points_placette[:, 2] < 640]
        # We do the same for the intensity
        if las_file == "POINT_OBS8.las":
            points_placette = points_placette[points_placette[:, -2] < 32768]
        if las_file == "Releve_Lidar_F39.las":
            points_placette = points_placette[points_placette[:, -2] < 20000]

        # We directly substract z_min at local level
        xyz = points_placette[:, :3]
        knn = NearestNeighbors(500, algorithm='kd_tree').fit(xyz[:, :2])
        _, neigh = knn.radius_neighbors(xyz[:, :2], 0.5)
        z = xyz[:, 2]
        zmin_neigh = []
        for n in range(len(z)):
            zmin_neigh.append(np.min(z[neigh[n]]))
        points_placette[:, 2] = points_placette[:, 2] - zmin_neigh

        all_points = np.append(all_points, points_placette, axis=0)
        dataset[os.path.splitext(las_file)[0]] = points_placette
        mean_dataset[os.path.splitext(las_file)[0]] = [np.mean(x_las) / 100, np.mean(y_las) / 100]

    return all_points, dataset, mean_dataset