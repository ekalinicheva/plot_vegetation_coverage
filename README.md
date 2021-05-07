# PointNet-based model for the prediction of vegetation coverage using 3D LiDAR point clouds

PyTorch implementation of a weakly supervised algorithm for the prediction of vegetation coverage of different stratum. The algorithm is based on PointNet model [ref] for 3D data classification and segmentation.
First, our algorithm compute the pointwise-predictions a point belong to one of 4 classes:
- low vegetation
- bare soil
- medium vegetation
- high vegetation

Then it reprojects each point to the corresponding vegetation stratum by using posterior probabilities.
Finally, we compute vegetation ratio for each stratum to get final results.

![](exemples_images/3_stratum.png)

### Example usage
We show how to use the code to reproduce the results in the notebook `notebook_demo.ipynb`. 
The notebook can also be directly run on [this google colab](https://colab.research.google.com/drive/1MoX46KhSgkyQ36uSi04OVJ3RVHw-SeDH#scrollTo=_jH5pCLHuAza).
