---
title: Image Classifier - Photo shot indoor or outdoor
emoji: 👁
colorFrom: green
colorTo: red
sdk: streamlit
sdk_version: 1.42.2
app_file: app.py
pinned: false
license: mit
short_description: Classify an image as indoor or outdoor
---

# Indoor/Outdoor Classifier

This simple classifier can identify if an image is take indoors or outdoors. This is a model developed to test a hypothesis that such a classification can be made using 3 things:
* very limited data set
* transfer learning from a small model
* very quick inference to keep deployment costs low

## Performance

In the training that I completed for this model, I have the following:
* Training set = 65 images (around 32 images for each label)
* Validation set = 16 images (8 images for each label)
* Testing set = 19 images (9 images for each label)

This is definitely sub-optimal. However, I want to use this to test my hypothesis. So this is good enough as a first iteration. 

## Pre-Trained Model

For this use-case, I am using the pre-trained model [MobileNetV2](https://huggingface.co/docs/transformers/model_doc/mobilenet_v2). In my use case, I have just 50 images each for the 2 labels that I will be training. Due to the small data set, a smaller model like MobileNet would be better suited.

## Model 
This model is currently hosted on HuggingFace at [akshayranganath/indoor-outdoor](https://huggingface.co/akshayranganath/indoor-outdoor).

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
