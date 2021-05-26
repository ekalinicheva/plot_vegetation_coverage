import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def augment(cloud_data):
    """augmentation function
    Does random rotation around z axis and adds Gaussian noise to all the features, except z and return number
    """
    # random rotation around the Z axis
    # angle = random angle 0..2pi
    angle = np.radians(np.random.choice(360, 1)[0])
    c, s = np.cos(angle), np.sin(angle)
    M = np.array(((c, -s), (s, c)))  # rotation matrix around axis z with angle "angle"
    cloud_data[:2] = np.dot(cloud_data[:2].T, M).T  # perform the rotation efficiently

    # random gaussian noise everywhere except z and return number
    sigma, clip = 0.01, 0.03
    cloud_data[:2] = (
        cloud_data[:2]
        + np.clip(
            sigma * np.random.randn(cloud_data[:2].shape[0], cloud_data[:2].shape[1]),
            a_min=-clip,
            a_max=clip,
        ).astype(np.float32)
    )
    cloud_data[3:8] = (
        cloud_data[3:8]
        + np.clip(
            sigma * np.random.randn(cloud_data[3:8].shape[0], cloud_data[3:8].shape[1]),
            a_min=-clip,
            a_max=clip,
        ).astype(np.float32)
    )
    return cloud_data


def cloud_loader(plot_id, dataset, df_gt, train, args):
    """
    load a plot and returns points features (normalized xyz + features) and
    ground truth
    INPUT:
    tile_name = string, name of the tile
    train = int, train = 1 iff in the train set
    OUTPUT
    cloud_data, [n x 4] float Tensor containing points coordinates and intensity
    labels, [n] long int Tensor, containing the points semantic labels
    """
    cloud_data = np.array(dataset[plot_id]).transpose()
    gt = (
        df_gt[df_gt["Name"] == plot_id][
            ["COUV_BASSE", "COUV_SOL", "COUV_INTER", "COUV_HAUTE", "ADM"]
        ].values
        / 100
    )
    # gt = np.asarray([np.append(gt, [1 - gt[:, 2]])])

    xmean, ymean = np.mean(cloud_data[0:2], axis=1)

    # normalizing data
    # Z data was already partially normalized during loading
    cloud_data[0] = (cloud_data[0] - xmean) / 10  # x
    cloud_data[1] = (cloud_data[1] - ymean) / 10  # y
    cloud_data[2] = (cloud_data[2]) / args.z_max  # z

    colors_max = 65536
    cloud_data[3:7] = cloud_data[3:7] / colors_max
    int_max = 32768
    cloud_data[7] = cloud_data[7] / int_max
    cloud_data[8] = (cloud_data[8] - 1) / (7 - 1)

    if train:
        cloud_data = augment(cloud_data)

    cloud_data = torch.from_numpy(cloud_data)
    gt = torch.from_numpy(gt).float()
    return cloud_data, gt


def cloud_collate(batch):
    """Collates a list of dataset samples into a batch list for clouds
    and a single array for labels
    This function is necessary to implement because the clouds have different sizes (unlike for images)
    """
    clouds, labels = list(zip(*batch))
    labels = torch.cat(labels, 0)
    return clouds, labels
