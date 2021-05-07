# PointNet-based model for the prediction of vegetation coverage using 3D LiDAR point clouds

PyTorch implementation of a weakly supervised algorithm for the prediction of vegetation coverage of different stratum. The algorithm is based on PointNet model [ref] for 3D data classification and segmentation.
First, our algorithm compute the pointwise-predictions a point belong to one of 4 classes:
- low vegetation
- bare soil
- medium vegetation
- high vegetation
Then it reprojects each point to the corresponding vegetation stratum by using posterior probabilities.
Finally, we compute vegetation ratio for each stratum to get final results.
