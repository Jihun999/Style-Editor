# TEXTOC:  Text-driven Object-Centric Style Transfer
<!-- ## [<a href="https://text2live.github.io/" target="_blank">Project Page</a>] -->

![teaser](teaser_image.png)


[//]: # (### Abstract)
>We present Text-driven Object-Centric Style Transfer (TEXTOC), a novel method that guides style transfer at an object-centric level using textual inputs. The core of TEXTOC is our Patchwise Co-Directional (PCD) loss, meticulously designed for precise object-centric transformations that are closely aligned with the input text. This loss combines a patch directional loss for textguided style direction and a patch distribution consistency loss for even CLIP embedding distribution across object regions. It ensures a seamless and harmonious style transfer across object regions. Key to our method are the Text-Matched Patch Selection (TMPS) and Pre-fixed Region Selection (PRS) modules for identifying object locations via text, eliminating the need for segmentation masks. Lastly, we introduce an Adaptive Background Preservation (ABP) loss to maintain the original style and structural essence of the image’s background. This loss is applied to dynamically identified background areas. Extensive experiments underline the effectiveness of our approach in creating visually coherent and textually aligned style transfers.

## Getting Started
### Installation

```
git clone https://github.com/qjwiflsdkf/TEXTOC_official.git
conda create --name TEXTOC python=3.8
conda activate TEXTOC 
pip install -r requirements.txt
```

### Run examples 
* Our method is designed to change textures of existing objects. It is not designed for adding new objects or significantly deviating from the original spatial layout.
* Training **TEXTOC** multiple times with the same inputs can lead to slightly different results.

The required GPU memory depends on the input image size.
In this project with set the image size as 512 x 512.

#### Image style transfer
Run the following command to start training
```
bash train.sh
```
Results will be saved to `model_output`. 

For more see the [Project page](https://anonymous.4open.science/w/textoc-A721/).
