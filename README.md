# Communications-Oriented-Token-Space-Analysis-and-Progressive-Compression-for-Vision-Language-Models
Test code for articles submitted to IEEE Transactions on Communications

This repository is built on the Communications-Oriented Token Space Analysisand Progressive Compression for Vision-Language Models

## Requirements
Dependencies can be found in requirements.txt

## Download the pre-trained codebook or model from the link below and place them in the designated directory: 
Pre-trained BLIP weights and BERT need to be downloaded before running. [BLIP]: (https://github.com/salesforce/BLIP) #In this paper, the BLIP w/ ViT-B is used.
[BERT]:(https://drive.google.com/drive/folders/1pZAzqDwhJMvo_N9JPB1pANfjMkNlqfGe?usp=drive_link)
In order to run the feature coding for machine algorithm proposed in this paper, it is also necessary to download the codebook：
[codebook]:(https://drive.google.com/file/d/1jv3pt70uSgXHUaRnpFujRQKNYIpAdNC0/view?usp=drive_link)

## Data preparation
Please place your dataset files as follows before running any scripts:

flick/ ├── annotation/ │ └── <annotation files> └── flickr30k-images/ └── flickr30k-images/ └── <image files>

- **annotation/**: Contains all annotation files.
- **flickr30k-images/**: The main folder for the Flickr30k dataset images.
  - **flickr30k-images/**: Subfolder with the actual image files.

Make sure to update the dataset paths in your configuration or script parameters accordingly.
## Test hybird progressive token compression

To generate a caption for an image:

<pre> python static_image_caption_hybird.py  </pre> 

To perform image retrieval：
<pre> python replace_retrieval_codebook_bpp_hybird.py  </pre> 



