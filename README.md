# Land Use Land Cover Classification and Map Generation

### Abstract 

Satellite imaging has many uses, and it is present in all facets of daily life. In particular, landscape images have advanced over time to address numerous issues in various fields. Hyperspectral remote sensors are frequently employed in remote sensing to monitor the surface of the planet with a high spectral resolution which results in better image generation. Utilizing remote sensing technologies is a useful approach to keep an eye on Earth's developments. LULC images have been retrieved via satellite imagery on a large scale.

In recent years, the creation of LULC images has become more significant for studies pertaining to climate change, landscape ecology, and sustainable land management.
Furthermore, data scientists state that temporal variations in LULC provide information regarding the appropriate use and planning of natural resources as well as their management. Therefore, current and accurate LULC information is essential for maintaining a sustainable ecosystem. In addition, as uncontrolled and irregular urban expansion can alter the urban climate, it is crucial to periodically monitor changes in the local urban light-curriculum in rapidly expanding cities.


### Problem Statement

Despite the increase in need for LULC classification models, many of the maps and digital databases that are currently in use were not created with the diverse needs of users in mind. Although it is typically overlooked, one of the primary causes is the kind of classification that is applied to fundamental data like land cover and land use. While there are numerous worldwide categorization schemes, there isn't a single internationally recognized system for classifying land use or cover. Here’s why researchers are focused on preparing a more accurate LULC classification model-

- 1. Diverse User Requirements
Information on land cover and land use is needed by a variety of stakeholders, including researchers, legislators, urban planners, and environmentalists. Databases and maps that are currently in use might not always meet these various needs[2].

- 2. Classification and Legend Issues
The classification scheme and legend chosen to characterize land use and cover can have a big influence on how relevant and comparable geographical data are. Cross-regional or cross-project data comparison can be hampered by inconsistent classification schemes.

- 3. Project-Oriented Approaches
A large number of the land cover and land use classifications that are currently in use were created for particular projects or industries, which restricts their general application. This may result in a lack of uniformity and fragmentation in the representation of land cover and land use data.

- 4. Lack of International Standardization
There is no internationally recognized standard for classifying land use or cover, despite efforts to create a number of classification schemes across the globe. This lack of uniformity may make it more difficult for nations and regions to share data and interact with one another.[5]

### Methodology 

The primary aim for the project is to manipulate, analyze, and visualize geospatial data and to get a deeper understanding on how the LULC map is generated for an area of Interest. In order to achieve this, the project could be divided into two parts- Fine Tuning a Pre-trained model for Image classification and then using the pre-trained model to generate a LULC Map for a particular region.

The pre-trained model used is ResNet50. ResNet-50 is a convolutional neural network architecture that belongs to the family of ResNet (Residual Network) models. ResNet-50 is a specific variant of ResNet that consists of 50 layers, including convolutional layers, pooling layers, and fully connected layers. It was introduced in the paper titled "Deep Residual Learning for Image Recognition" in 2015[6]

Deep neural networks are difficult to train due to the problem of vanishing or exploding gradients (repeated multiplication making the gradient infinitely small). ResNet bypasses one or more levels in order to overcome this by creating shortcut connections, as seen below, that link activation from an earlier layer to a later one.[6].

#### 1. Fine-tuning the ResNet50 Model
Fine-tuning a pretrained model refers to the process of taking a neural network model that has been trained on a large dataset (typically a general dataset like ImageNet) and further training it on a new, smaller dataset specific to a particular task or domain. The goal of fine-tuning is to take advantage of the knowledge that the pretrained model (which was trained on a sizable dataset) has acquired and modify it so that it can function well on a new task or dataset. Below is the sequence of fine-tuning done for the ResNet50 model

- <u>Creating Custom dataset classes<u>
By Creating custom dataset classes In Pytorch the Dataset class allows you to define a custom class to load the input and target for a dataset. this capability is used to load the input in the form of RGB satellite images along with their labels and later apply any kind of transformations
- ii. <u>Data Augmentation<u>
During model training, data augmentation involves randomly applying image changes, such as cropping, flips (horizontal and vertical), and other adjustments, to the input images. The neural network can more effectively generalize to the unknown test dataset thanks to these perturbations, which also lessen the network's overfitting to the training dataset.

- iii. <u>Image Normalization<u>
Image normalization is a preprocessing method used to standardize an image's pixel values in computer vision and image processing. By ensuring that the input data (image pixels) have a constant scale and distribution, picture normalization aims to enhance the stability and performance of machine learning models.


#### 2. Generating Land Use and Land Cover Map for an Area

Sentinel-2 satellite image for a region of interest is downloaded through Google Earth Engine and a trained CNN model(ResNet50) is applied on the image to generate a land use and land cover map.
- i. Generating sentinel-2 Satellite images
Sentinel-2 is a Copernicus Program Earth observation project that produces global multispectral imagery at 10 m resolution every 10 days from 2015 to the present[1].A function is written to use the Python Earth Engine API to create a Sentinel-2 image from Google Earth. An aggregator is selected for a set of photographs taken over time rather than a single image on a specific date in order to reduce cloud cover.
- ii. Visualization of the Sentinel-2A Image
Once the API  call is done and the image is received , a boundary is made to differentiate states(of U.S.A) and a portion of a state- California is selected as a region to generate the LULC map for. After the region is downloaded, it is visualized as a satellite raster image using the raster library. Pixels are arranged in a grid to form raster data. Digital elevation maps, maps of nocturnal luminosity, and multispectral satellite photos are a few examples. For example, red, green, and blue values in satellite pictures; night light intensity in NTL maps; and height in elevation maps—each pixel denotes a value or class. GeoTIFFs (.tiff) are frequently used to store raster data.
- iii. Generating & Visualizing 64x64 px GeoJSON Tiles
Since the deep learning model trained on RGB dataset was trained on 64*64 pixel Sentinel-2 image patches, the generated Sentinel-2 image from google earth engine need to be broken down as well into smaller 64*64 patch tiles. The is done by creating a function that generates a grid of 64*64 tiles using 'Rasterio Window Utilities'. Pixels are arranged in a grid to form raster data. The land use and land cover classification map is generated using the trained model. The RGB Dataset consists of 10 different LULC classes which will be explained further in the data section. Later, an interactive LULC map is generated using the Folium library after loading the resulting predictions made earlier.

By combining all the above steps, the finely-tuned ResNet-50 model is leveraged to create accurate and detailed LULC maps for various environmental and land management applications. The key is to adapt the deep learning model to the specific characteristics and requirements of the target area and land cover classification task.

##### 3. Algorithms
Basic Algorithm structure Combining the above mentioned two parts
- Step1- Preparation of Data
Gather satellite imagery data that could be RGB or Multi-spetural captured by satellites like Sentinel-2 or Landsat and acquire ground truth data or labeled samples for training and validation.
- Step2- Load Pre-Trained ResNet-50 Model
Import the ResNet-50 model architecture along with pre-trained weights through Deep learning networks like pytorch
- Step3- Modify and Fine-Tune Model
Training the modified model using the RGB dataset and feeding the training images into the model for optimizing weights and minimizing the loss function (e.g., cross-entropy) 
- Step4- Validation and Evaluation
Evaluate the fine-tuned model using a validation dataset to assess its performance in classifying LULC types
- Step5- Generate LULC Map
Generation of a pixel-wise LULC map based on the model's predictions where each pixel in the map represents a specific land cover or land use class
- Step6- Post-Processing and Visualization
Visualize the generated LULC map using mapping libraries (e.g.,  Folium) to visualize the distribution of different land cover and land use classes within the region of Interest.


