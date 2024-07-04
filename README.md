# Computer-Aided Diagnosis of Pituitary Microadenoma on Dynamic Contrast-Enhanced MRI Based on Spatio-Temporal Features
Te Guo, Jixin Luan, Jingyuan Gao, Bing Liu, Tianyu Shen, Hongwei Yu, Guolin Ma∗, Kunfeng Wang∗

∗Corresponding authors

### Framework Overview
Overall framework of the proposed intelligent diagnostic model for PM, including a classification module, pituitary segmentation module, and PM segmentation module. This paper utilize an optimized model based on ZFNet to adequately extract target semantic information while preserving clear spatial details. This paper introduce a DSSM and a RUIM to further capture more precise semantic information, thereby reducing precision loss and enhancing the utilization of low-level information for improved detection of small objects.
![image](https://github.com/BUCT-IUSRC/Research__PM-CAD/assets/58768104/93205d66-ac48-4212-a165-ef966fe6a9ff)


### Instal
### Environment
```sh
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
```


### Run

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

### Contact Us
If you have any problem about this work, please feel free to reach us out at 2022400201@buct.edu.com
