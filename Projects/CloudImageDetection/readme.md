# Cloud Image Detection project 

The cloud image detection project is a full blown deep learning project done as a part of my Data Science journey. Being concerned about climate change, the goal of this project is to be able to create an image detection network which can accurately classify various cloud patterns. The cloud images have been scraped from the internet using scraping agents in order to collect images on cloud patterns. 

Due to most of the images not being labelled, this project will focus on a **semi-supervised learning** approach, with a 3-step phase process for working around the image labelling problem:

* Scrape the images from various web sources and produce a bundle of these images 
* Manually label a few cloud pattern images for each respective cloud pattern as to make a training batch 
* Approach the labelling problem via the ResNet approach implemented in this paper https://arxiv.org/abs/1905.00546 [Implementing Billion-scale semi-supervised learning for image classification]
* Once labelling has been performed, prepare the images for CNN training and validation procedure. 
* Test the final network on newly procured cloud images and test its accuracy under various image settings. 
