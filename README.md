# Movie poster classification
A project where a CNN based pipeline is set up to predict the genre of a movie based on the poster image. 

Group 20
- Thije Robbe (s4578961)
- Lennart Centen (s3801829)
- Bonne Jongsma (s3501647)
- Stijn de Vries (s3447146)

## Pre-processing
- ``dimension-graph.py`` - Plot the distribution of height pixels (rows) against the percentage of the images that has an equal or larger number of pixels. Used for selecting an appropriate value which would retain enough information in the image while also retaining a large enough dataset size.
- ``get_metadata.py`` - Add image metadata to dataset csv file
- ``insights.py`` - print simple statistics and information of the dataset.
- ``preprocessing.py`` - Generate a local image dataset by downloading and resizing the images indicated by the url in the csv file.

## Feature extraction
- ``color_descriptors.py`` - Simple pipeline to evaluate the usability of the distribution of colors within an image as a input vector to a classifier.
- ``pca.py`` - Incomplete experiment using PCA as dimension reduction.

## Classification
- ``CNN.py`` - Architecture of the CNN used as classifier in this project, used in ``CNN_TEST.py``.
- ``data_generator.py`` - Dataset object to enable batch-processing of the data, preventing having to load the entire image dataset into memory.
- ``CNN_TEST.py`` - CNN classifier with K-fold cross validation
- ``vgg_transfer_learning.py`` - CNN classifier using the pre-trained VGG_16 model.

## Data
- 41K_processed.csv - Dataset containing the URLs to the images, their classes and other metadata.

Note: The entire image dataset is to large to be included in the repository. The images can be found on the Nestor file exchange for group 20. The contents of the zip file ``normal.zip`` should be extracted to the folder ``data/normal/``. 

## Models
This folder contains copies of the models trained during the project.

# Running the code
Any of the python programs can be run from the root of the repository with the command ``python3 classification/CNN_TEST.py``. It is important to run the code from the root of the repository, because the code locates files with relative links.

# Required python packages
- scikit-learn
- scikit-image
- opencv-python
- numpy
- pandas
- torch
- PIL
- matplotlib