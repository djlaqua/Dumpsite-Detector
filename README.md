# Dumpsite-Detector

# Illegal Waste Detection using Faster R-CNN


### Introduction

Illegal waste dumping poses significant environmental and health risks. Identifying and cleaning up can significantly reduce soil, water, and air pollution. These sites often contain hazardous materials that can leach into the ground and contaminate water sources, harm wildlife, and affect the surrounding ecosystem. People living near illegal dumpsites are at risk of exposure to hazardous substances, which can lead to serious health issues like respiratory problems, skin irritations, and even cancer. Detecting these sites helps in minimizing these risks.The process of detecting and cleaning up illegal dumpsites can be resource-intensive and costly. This project aims to leverage computer vision techniques to automatically detect and locate dumpsites, enabling prompt intervention and responsible waste management.


### Problem Statement

This project focuses on the detection and localization of illegal waste dumpsites using the Faster R-CNN (Region-based Convolutional Neural Network) approach. The goal is to provide an efficient and accurate solution for identifying areas with unauthorized waste disposal.

## Project Objectives
<ul>
<li>Collect and curate a comprehensive dataset of images that includes a variety of dumpsites, including image annotation and preprocessing.</li>

<li>Design and implement an Faster R-CNN based model to accurately detect illegal waste dumpsites.
<ul>
<li>Utilizes the Faster R-CNN algorithm for object detection.</li>
 <li>Precisely identifies the position of illegal waste dumpsites within images.</li>
</ul>
</li>

<li>Conduct thorough training of the model using the prepared dataset</li>

<li>Validate the model using separate test data to evaluate its performance and accuracy.</li>
</ul>

###  Data Scope

In this project, I leverage two distinct datasets to address the challenge of illegal waste detection from satellite imagery. Our primary dataset originates from the Global Dumpsite Test Data https://www.scidb.cn/en/s/6bq2M3, which is a comprehensive compilation derived from various cities worldwide, including Colombo in Sri Lanka, Dhaka in Bangladesh, Guwahati in India, Kinshasa in the Democratic Republic of Congo, Lagos in Nigeria, New Delhi in India, and several cities in China. The images within this dataset are of size 1024 × 1024 pixels and have been meticulously labeled, classifying dumpsites into categories such as domestic waste, construction waste, agricultural waste, and covered waste. For our specific project, I aggregated these diverse classifications into a singular "dumpsite" class. The dataset is well-structured and provides a valuable resource for training our model on a variety of waste scenarios.

As the second source of data, I turn to the AerialWaste dataset https://aerialwaste.org/, specifically curated for the discovery of illegal landfills. This dataset captures the visual heterogeneity of scenes featuring waste dumps in aerial images, presenting a diverse array of objects within waste deposits. In this context, I consider non-dumpsite images from the AerialWaste dataset to constitute our second class. Effectively, I frame our problem as a binary classification task, distinguishing between images containing illegal waste dumpsites and those representing other scenes within the realm of waste disposal.

By combining these two datasets, I enhance the robustness of our model, ensuring its effectiveness in identifying and localizing illegal waste dumpsites from varying perspectives and contexts.

The model produced great results in detected dumpsites across multiple Geographic regions

> <img src='https://github.com/djlaqua/Dumpsite-Detector/blob/main/data1/detect_imag.png' alt='?' style='width:900px'/>

Trained models will be pushed later on
