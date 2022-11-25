
# ASD classification using Large-margin Softmax Loss for stacked 2D fMRI data



Incorporated [CBAM](https://arxiv.org/pdf/1807.06521.pdf) with Resnet-50 and [L-margin softmax loss](https://arxiv.org/pdf/1612.02295)
to carry ASD classification on the [ABIDE-1](https://fcon_1000.projects.nitrc.org/indi/abide/) dataset fMRI data in [Pytorch](https://pytorch.org/).


## Dataset

Resting-state functional magnetic resonance imaging (fMRI) data is obtained
from [Autism Brain Imaging Exchange (ABIDE-1)](https://fcon_1000.projects.nitrc.org/indi/abide/). 

## Prepocessing steps:
__Derivative__: func_mask (3D Functional data mask)  
__Pipeline__ : CCS    
__Strategy__ : filt_global (band-pass filtering and global signal regression)


## Processing 
3D fMRI data is downloaded using the [ABIDE_3D.py](https://github.com/manugaur99/ASD-classification-using-Large-margin-Softmax-Loss-for-stacked-2D-fMRI-data/blob/main/ABIDE_3D.py) script.
We convert the 3D neuroimaging data to 2D brain images across the axial plane using med2image[https://github.com/FNNDSC/med2image].

The 2D images are normalized, augmented in unison and stacked volumetrically along the z-axis. 


## Model
We implement a Resnet-50 with CBAM module compatible with (64x64) and (32x32) images. We minimize the CrossEntropy loss and Large-margin softmax loss together. 
We also utilize batch-wise data mixing via [mixup](https://arxiv.org/pdf/1710.09412).

## Performance 

Test accuracy = 90.57, F1-score = 89.31, AUC  = 0.91


## Environment Variables

### Hardware requirements
A server containing CUDA enabled GPU with compute capability 3.5 or above.


### Software requirements

`Pytorch version 0.4.1`   

`CUDA version 8+`  

`Python version 3.5+` 

`med2image ` 




## Environment Variables

### Hardware requirements
A server containing CUDA enabled GPU with compute capability 3.5 or above.


### Software requirements

`Pytorch version 0.4.1`   

`CUDA version 8+`  

`Python version 3.5+` 

`Albumentations` 





## References

1. [CBAM attention module](https://arxiv.org/pdf/1807.06521.pdf).
2. [Large-margin softmax loss](https://arxiv.org/pdf/1612.02295).
3. [mixup](https://arxiv.org/pdf/1710.09412).
