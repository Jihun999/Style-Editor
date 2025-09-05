# Style-Editor: Text-driven Object-centric Style Editing (CVPR 2025 highlight)

Official implementation of Style-Editor: Text-driven Object-centric Style Editing

[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Park_Style-Editor_Text-driven_Object-centric_Style_Editing_CVPR_2025_paper.pdf) [Project page](https://jihun999.github.io/projects/Style-Editor/)  



![teaser](teaser_image.png)


[//]: # (### Abstract)
>We present Text-driven object-centric style editing model named Style-Editor, a novel method that guides style editing at an object-centric level using textual inputs.The core of Style-Editor is our Patch-wise Co-Directional (PCD) loss, meticulously designed for precise object-centric editing that are closely aligned with the input text. This loss combines a patch directional loss for text-guided style direction and a patch distribution consistency loss for even CLIP embedding distribution across object regions. It ensures a seamless and harmonious style editing across object regions.Key to our method are the Text-Matched Patch Selection (TMPS) and Pre-fixed Region Selection (PRS) modules for identifying object locations via text, eliminating the need for segmentation masks. Lastly, we introduce an Adaptive Background Preservation (ABP) loss to maintain the original style and structural essence of the image's background. This loss is applied to dynamically identified background areas.Extensive experiments underline the effectiveness of our approach in creating visually coherent and textually aligned style editing.


## Getting Started
### Installation

```
git clone https://github.com/Jihun999/Style-Editor.git
conda create --name style_editor python=3.9
conda activate style_editor 
pip install -r requirements.txt
```

### Run examples 
* Our method is designed to change style of existing objects by using text description. It is not designed for adding new objects or significantly deviating from the original spatial layout.
* Training **Style Editor** multiple times with the same inputs can lead to slightly different results.
* The source image, source text and style text are the input of our model.

The required GPU memory depends on the input image size.
In this project, we set the image size to **512 × 512**, which typically requires **more than 15 GB GPU memory**.

#### Image style transfer
Run the following command to start training
```
bash train.sh
```
Results will be saved to `model_output`. 

## Acknowledgement
This code is implemented on top of [CLIPstyler](https://github.com/cyclomon/CLIPstyler).

We thank the authors for open-sourcing great projects and papers!

## Citation
Please kindly cite our paper if you use our code and data:

```bibtex
@inproceedings{park2025style,
  title={Style-Editor: Text-driven object-centric style editing},
  author={Park, Jihun and Gim, Jongmin and Lee, Kyoungmin and Lee, Seunghun and Im, Sunghoon},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18281--18291},
  year={2025}
}
```
