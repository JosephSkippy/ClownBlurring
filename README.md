# Introduction

Scay clown images brings discomfort to children. By leveraging computer vision techniques and machine learning algorithms, we can detect and blur specific regions of an image or video that contain scary clown faces. This can be useful for creating a safer and more comfortable viewing experience for individuals who are sensitive to such content. 

# Executive Summary

I have used YOLOv8 nano model to perform instance segmentation of images containing clown. The data is scrapped using selenium and annotated using roboflow.

Only 200+ annotated are to perform for POC due to limited human resource to annotate.

Initial iteration performance as following.

| Iteration | Number of Training Image/Instance | Number of Validation Image/Instance | Number of Test Image/Instance | Epoch | metrics/mAP50 | metrics/mAP50-95 | metrics/mAP50(M)@Test | metrics/mAP50-95(M)@Test |
|-----------|-----------------------------------|-------------------------------------|-------------------------------|-------|---------------|------------------|-----------------------|--------------------------|
| 1         | 171                               | 34                                  | 23                            | 100   | 0.784         | 0.594            | 0.815                 | 0.611                    |


## Example where Image

Before

<img src="results/before.png" width="250" height="250">

After

<img src="results/after.png" width="250" height="250">

## Example where Video (Success)

<img src="results\clown_blurred_succes.gif" width="250" height="250">

## Example where Video (Failed)

<img src="results/clown_blurred_fail.gif" width="350" height="250">

## Observation

1. It is noticable that the images failed where the faces are small. Hence the data trained shall includes small faces

2. It is also noted that human faces are wrongly predicted as clown. The trained data shall include human class to teach the model to differentiate between human and clown.

## Future 

1. Include more training data with different distribution
2. deployment to cloud

