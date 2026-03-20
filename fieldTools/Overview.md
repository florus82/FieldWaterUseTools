# Overview of field delineation procedure
The goal is a 10m porduct for the entirety of Germany. We follow the approach and model by Diakogiannis et al.
https://essd.copernicus.org/articles/15/317/2023/essd-15-317-2023-discussion.html, https://github.com/feevos/tfcl, 


## Training of 3D Vision Transformer model based on AI4Boundaries dataset and/or Fine-Tuning dataset
model is based on 4 10m Sentinel-2 Bands

- fine-tuning with rasterized IACCS polygons with dilate=True
- fine-tuning with rasterized IACCS polygons with dilate=False
- from scratch with rasterized IACCS polygons with dilate=True (currently still running)
- from scratch with rasterized IACCS polygons with dilate=False
- maybe, from scratch with dilate=T, and fine-tune with dilate=F


## Results so far

<p align="center">
  <img src="model_results_comparison.png" width="1200">
</p>


## predicting the model

- applies the model to numpy array that is sliced into small chips that fit the model's dimensions. Furthermore, the chips do overlap, so that the merged predicted chips join more smoothly (the overlap will be cut of when storing predictions on disc)

A mask is applied optionally. We used so far the IACS masks which we cannot do for the entirety of Germany. 

    --> Hence, the Thüenen croptype map might be more adequate

- we are currently testing different combinations of chipsize, overlap and masking (with Thuenen)
 
## Validation procedure I (grid search)

- we cut the rasterized IACS dataset that matches the state and year of the prediction to the same extent as of the prediction
- we slice the vrt of a prediction (probability) into tiles (similar shape as FORCE tiles) for easier processing
- due to the slicing we neglect fields close to the resulting edges of tiles in order to exclude sliced fields from the validation process
- then, we draw randomly samples across deciles of fieldsizes (pixelcount), where every decile gets 10% of overall samplesize
- we test different parameter combinations (thresholds for probability maps for fields and boundaries) in a watershed segmentation algorithm
  --> thoughts on that: if parameter for probability for field is not chosen very high (>.8), there are hardly any boundaries. Furhermore, simply thresholding the extent probability map appears to deliver the same output as InstSegm
- in the tests, we compare against an IACS mask that is adapted with respective thuenen mask (where thuenen mask == 0 -> IACS = 0)


<p align="center">
  <img src="grid_search_AI4_RGB_exclude_True_38.png" width="800">
</p>


## Validation procedure II (after segmenting probability maps into final field map)

- use matching validation set from vault and test the map

## Validation procedure III (vectorized final field map vs IACS)

- how do we do that? all vs sample, scores...

## Questions
- should we really exclude vineyards, what about Brachen, Hopfen, 




