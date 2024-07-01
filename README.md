# Research__PM-CAD
### Dependencies
numpy
scipy
Pillow
cython
matplotlib
scikit-image
tensorflow>=2.2.0
opencv-python
h5py
imgaug
IPython[all]
### Train

To train the pituitary segmentation model:

```sh
cd segmentation
sh samples/pm/pm-seg.sh
```

To train the microadenomas diagnostic model :

```shell
cd diagnostic
sh pm-dia.py
```
