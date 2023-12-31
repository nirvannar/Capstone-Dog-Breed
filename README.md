# Capstone-Dog-Breed
Udacity Capstone Project

This GitHub repo contains your code (if possible), dataset (unless you chose the Spark project option), and README.md files.Capstone Project is the final project for the Data Science nanodegree.

# Table of Contents

* [Installation](#Installation)
* [Project Motivation](#Project-Motivation)
* [File Descriptions and Analyses](#File-Descriptions-and-Analyses)
* [Results](#Results)
* [Licensing, Authors, and Acknowledgements](#Licensing,-Authors,-and-Acknowledgements)


## Installation <a name="Installation"></a>
The code should run with no issues using Python versions 3.8.8 using Jupyter Notebook server version 6.3.0.  Numpy, sklearn.datasets, glob, keras.utils, sklearn.datasets, random, cv2, matplotlib.pyplot, keras.applications.resnet50, keras.preprocessing, tqdm, PIL, keras.layers, keras.models, keras.optimizers, keras.callbacks, and os were the libraries used.  

## Project Motivation <a name="Project-Motivation"></a>
This Capstone Project is the final project for the Data Scientist Nanodegree Program.  For this project given an image of a dog, the classifier housed in a Jupyter notebook is expected to identify the breed. If supplied an image of a human, the code will identify the resembling dog breed. Otherwise it will output an error message. 



**_Business Problem Understanding_**
The project will be divided into the following tasks

**Step 0: Import Datasets**
The data was explored with descriptive statistics and the number of ``train_files, valid_files, test_files`` and ``dog_names`` representing breeds were counted.

**Step 1: Detect Humans**
Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.

**Step 2: Detect Dogs**
A pre-trained ResNet-50 model to detect dogs in images was used.  The data was pre-processed.  Then a ``dog_detector`` function was then written, applied to the numpy arrays ``human_files_short`` and ``dog_files_short`` with the targeted output returning ``True`` if a dog is detected in an image and ``False`` if not detected.

**Step 3: Create a CNN to Classify Dog Breeds (from Scratch)**
Finally a machine learning approach to building recommendations was used. Using the user-item interactions,  a matrix decomposition was built. 

**Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)**
The task was to use transfer learning to create a CNN using VGG-16 bottleneck features.The model was compiled and trained with 20 epochs. The model with the best validation loss was savedusing model checkpointing and then  loaded and tested to gauge how well it identifies the dog breeds. The achieved accuracy was 49.64%.

**Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)**
The bottleneck features from a downloaded ResNet50 pre-trained model were used in Section 5. The required target accuracy from the CNN was to be at least 60% on the test set.The model was compiled and trained with 20 epochs. Model checkpointing was used to save the model that attained the best validation loss. The achieved test accuracy was 80.98%.

**Step 6: Write your Algorithm**

**Step 7: Test Your Algorithm**


## File Descriptions and Analyses <a name="File-Descriptions-and-Analyses"></a>
**_Data Preparation_**  
'data/user-item-interactions.csv' and 'data/articles_community.csv' were explored.

``haarcascades``: folder contains pre-trained face detectors

haarcascade_frontface_alt.xml: the pre-trained face detector used in the dog_app.ipynb notebook
images: folder contains all example images shown in the dog_app.ipynb notebook

test: folder contains all test images used to test the algorithm in step 7

dog_app.ipynb: Jupyter notebook that contains full codes to build up CNNs and algorithm

extract_bottleneck_features.py: Python function to extract bottleneck features

bottleneck_features: folder contains pre-trained models
Download the VGG-16 bottleneck features for the dog dataset. Place it in the repo, at location path/to/dog-project/bottleneck_features.

DogVGG16Data.npz: pre-trained VGG16 model (this has already been downloaded)
DogResnet50Data.npz: pre-trained ResNet-50 model (this will need to be downloaded from the link provided above)
saved_models: folder contains trained models from the dog_app.ipynb notebook (all are saved while running the dog_app.ipynb)

weights.best.from_scratch.hdf5: stores the trained model with the best validation loss from self-created CNN
weights.best.VGG16.hdf5: stores the trained model with the best validation loss from VGG16 model
weights.bes.Resnet50.hdf5: stores the trained model with the best validation loss from ResNet-50 model


## Results <a name="Results"></a>

**I. Exploratory Data Analysis**
![image](https://github.com/nirvannar/Capstone-Dog-Breed/assets/52913504/1b9a8581-4a08-4407-96c3-4335685824bf)


**Step 0: Import Datasets**
There are 133 total dog categories.
There are 8351 total dog images.
There are 6680 training dog images.
There are 835 validation dog images.
There are 836 test dog images.

There are 13233 total human images.

**Step 1: Detect Humans**
A human face detector was written and assessed. 100 % of the first 100 images in human_files have a detected human face while 11 % of the first 100 images in dog_files have a detected human face.  

**Step 2: Detect Dogs**
A ``dog_detector`` function was then written and assessed with the desired output.  0 % of the images in human_files_short have detected a dog.  100 % of the images in dog_file_short have detected a dog

**Step 3: Create a CNN to Classify Dog Breeds (from Scratch)**
Finally a machine learning approach to building recommendations was used. Using the user-item interactions,  a matrix decomposition was built. 

**Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)**
I used transfer learning to create a CNN using VGG-16 bottleneck features.

**Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)**


**Step 6: Write your Algorithm**


**Step 7: Test Your Algorithm**


The SVD in the lesson did not converge as it had encountered a NaN or missing values in the user_movie_subset matrix. There was a significant missing data. "The number of ratings made for user-movie pairs that didn't have ratings is 13 835 713". This means it was a very sparse matrix.  SVD from NumPy does not work when the matrices do not have a value in every cell and is thus considered not complete. FunkSVD which uses a stochastic gradient descent formula was used in the lesson to address NaN/missing values.  SVD in this Project is used on the user_item_matrix which has no missing values.

The training data suggests that as latent features increase the accuracy increases. However the test data suggests that accuracy decreased as latent features increased. There is direct conflict between the two. This difference could be due to the small overlap of users between training and testing datasets. Only 20 users in the test dataset are shared with the training dataset and were used to give the result.

We could increase the sizes of the test datasets, use online recommendation evaluations such as A/B testing, or time-based testing as opposed to matrix factorization used above which is an offline method.

All types of recommendations should be used not just user-user based collaborative and rank-based recommendations. Content-based like that desctibed in Part III question (6) for new users and knowledge-based recommendations for existing could be used.


[Medium blog](https://medium.com/@nirvannsramp/dog-breed-classifier-b485852f6b0d)


## Licensing, Authors, and Acknowledgements<a name="Licensing,-Authors,- and- Acknowledgements"></a>

[Labrador Retriever image](https://hips.hearstapps.com/hmg-prod/images/dog-1598970334.png?crop=0.563xw:1.00xh;0.231xw,0&resize=980:*)

[Chow chow dogs image](https://hips.hearstapps.com/hmg-prod/images/chow-chow-dog-breed-651d88c291c70.jpg?crop=0.668xw:1.00xh;0.197xw,0&resize=1200:*)


[Algoscale](https://algoscale.com/blog/yolo-vs-ssd-which-one-is-a-superior-algorithm/#:~:text=The%20main%20difference%20between%20YOLO,score%20of%20more%20than%200.5.)

[dlib.com](http://dlib.net/imaging.html)

[techtarget.com](https://www.techtarget.com/searchenterpriseai/definition/face-detection) 

[faceonlive.com](https://faceonlive.com/face-detection-models-the-ultimate-guide-2023-unleash-the-power-of-ai-to-spot-faces-like-a-pro/#:~:text=Dlib's%20CNN%3A%20Ideal%20for%20applications,recognition%20systems%20or%20video%20surveillance.)

[Intel](https://www.intel.com/content/www/us/en/internet-of-things/computer-vision/convolutional-neural-networks.html#:~:text=A%20CNN%20model%20is%20a,training%2C%20optimizing%2C%20and%20inference.)

[GeeksforGeeks](https://www.geeksforgeeks.org/ml-machine-learning/)

[ResearchGate](https://www.researchgate.net/figure/Relationship-between-AI-ML-DL-and-CNN_fig1_358745776)

[CNN transfer learning](https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce)

[Ai, ML and DL differences](https://towardsdatascience.com/understanding-the-difference-between-ai-ml-and-dl-cceb63252a6c)








