# Communications-Oriented-Token-Space-Analysis-and-Progressive-Compression-for-Vision-Language-Models
Test code for articles submitted to IEEE Transactions on Communications

This repository is built on the Communications-Oriented Token Space Analysisand Progressive Compression for Vision-Language Models

## Requirements
Dependencies can be found in requirements.txt
## matters needing attention
Pre-trained BLIP weights and BERT need to be downloaded before running. [BLIP]: (https://github.com/salesforce/BLIP) #In this paper, the BLIP w/ ViT-B is used.
[BERT]:(https://drive.google.com/drive/folders/1pZAzqDwhJMvo_N9JPB1pANfjMkNlqfGe?usp=drive_link)
## Test hybird progressive token compression

To generate a caption for an image:

<pre> python static_image_caption_hybird.py  </pre> 

To perform image retrieval, run the replace_retrieval_codebook_bpp_hybird.py

## Download the pre-trained weights from the link below and place them in the designated directory: 

