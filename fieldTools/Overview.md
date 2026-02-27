# Overview of field delineation procedure
The goal is a 10m porduct for the entirety of Germany. We follow the approach and model by Diakogiannis et al.
https://essd.copernicus.org/articles/15/317/2023/essd-15-317-2023-discussion.html, https://github.com/feevos/tfcl, 


## Training of 3D Vision Transformer model based on AI4Boundaries dataset and/or Fine-Tuning dataset
model is based on 4 10m Sentinel-2 Bands

- fine-tuning with rasterized IACCS polygons with dilate=True
- fine-tuning with rasterized IACCS polygons with dilate=False
- from scratch with rasterized IACCS polygons with dilate=True
- from scratch with rasterized IACCS polygons with dilate=False


## Results so far

<p align="center">
  <img src="model_results_comparison.png" width="800">
</p>


## predicting the model

- applies the model to numpy array tha is sliced into small chips that fit the model's dimensions. Furthermore, the chips do overlap, so that the model can predict better?? This overlap will be cut of when storing predictions on disc.

A mask is applied optionally. We used so far the IACS masks which we cannot do for the entirety of Germany. 

    --> Hence, the Thüenen croptype map might be more adequate

 
## Validation procedure



## To Do!

- when predicting, test larger overlaps
- test the Thüenen Agro-Maps for masking

## Questions
- should we really exclude vineyards, what about Brachen, Hopfen, 




