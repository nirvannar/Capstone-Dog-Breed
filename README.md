# Capstone-Dog-Breed
Udacity Capstone Project for the Data Scientist Nanodegree Program

This GitHub repo contains the code, dataset, and README.md files for the Udacity Capstone Project is the final project for the Data Science nanodegree.

# Table of Contents

* [Installation](#Installation)
* [Project Motivation](#Project-Motivation)
* [File Descriptions and Analyses](#File-Descriptions-and-Analyses)
* [Results](#Results)
* [Licensing, Authors, and Acknowledgements](#Licensing,-Authors,-and-Acknowledgements)


## Installation <a name="Installation"></a>
The code should run with no issues using Python versions 3.8.8 and using Jupyter Notebook server version 6.3.0.  
Numpy, sklearn.datasets, glob, keras.utils, sklearn.datasets, random, cv2, matplotlib.pyplot, keras.applications.resnet50, keras.preprocessing, tqdm, PIL, keras.layers, keras.models, keras.optimizers, keras.callbacks, and os were the libraries used.  

## Project Motivation <a name="Project-Motivation"></a>
This Capstone Project is the final project for the Data Scientist Nanodegree Program.  For this project given an image of a dog, the classifier housed in a Jupyter notebook is expected to identify the breed. If supplied an image of a human, the code will identify the resembling dog breed. Otherwise it will output an error message.  

**_Business Problem Understanding_**  
The project will be divided into the following tasks

**Step 0: Import Datasets**  
The data was explored with descriptive statistics and the number of ``train_files, valid_files, test_files`` and ``dog_names`` representing breeds were counted.

**Step 1: Detect Humans**  
Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.

**Step 2: Detect Dogs**  
A pre-trained ResNet50 model to detect dogs in images was used.  The data was pre-processed.  Then a ``dog_detector`` function was then written, applied to the numpy arrays ``human_files_short`` and ``dog_files_short`` with the targeted output returning ``True`` if a dog is detected in an image and ``False`` if not detected.

**Step 3: Create a CNN to Classify Dog Breeds (from Scratch)**  
Step 3 required that I create a CNN from scratch that classifies dog breeds. The required test accuracy to be attained was to be at least 1%.

**Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)**

The task was to use transfer learning to train a CNN using VGG-16 bottleneck features.The model was compiled and trained with 20 epochs. The model with the best validation loss was saved using model checkpointing.  The model was loaded and tested to gauge how well it identifies the dog breeds. 

**Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)**  
The task was to use transfer learning to create a CNN. The bottleneck features from a downloaded ResNet50 pre-trained model were used in Section 5. The required target accuracy from the CNN was to be at least 60% on the test set.The model was compiled and trained with 20 epochs. Model checkpointing was used to save the model that attained the best validation loss. 

**Step 6: Write your Algorithm**  
An algorithm was written that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. If a dog is detected in the image, it returns the predicted breed. If a human is detected in the image, it returns the resembling dog breed. If neither is detected in the image, an error message is displayed.
The face_detector and dog_detector functions developed in steps 1 and 2 above were used. The CNN from Step 5 was used to predict the dog breed.

**Step 7: Test Your Algorithm**  
The algorithm was to be tested on at least two human and two dog images.  


## File Descriptions and Analyses <a name="File-Descriptions-and-Analyses"></a>
**_Data Preparation_**  


``haarcascades``: folder contains pre-trained face detectors

haarcascade_frontface_alt.xml: the pre-trained face detector used in the dog_app.ipynb Jupyter notebook

images: folder contains example images used to test in step 7

test: folder contains additional test images used to test in step 7

dog_app.ipynb: Jupyter notebook containing the Dog Breed Workspace

extract_bottleneck_features.py: Python function to extract bottleneck features

bottleneck_features: folder contains pre-trained models

DogVGG16Data.npz: pre-trained VGG16 model   

DogResnet50Data.npz: pre-trained ResNet50 model   

saved_models: folder stores the trained models with the best validation loss for steps 3 to 5

weights.best.from_scratch.hdf5: stores the trained model with the best validation loss from self-created CNN

weights.best.VGG16.hdf5: stores the trained model with the best validation loss from VGG16 model

weights.bes.Resnet50.hdf5: stores the trained model with the best validation loss from ResNet50 model


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
The model was compiled and trained. The model with the best validation loss accuracy was loaded and then tested to the required accuracy. The achieved test accuracy was 3.25%. 

**Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)**
I used transfer learning to create a CNN using VGG-16 bottleneck features.The achieved accuracy was 49.64%.

**Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)**
I used transfer learning to create a CNN using ResNet50 bottleneck features.The achieved test accuracy was 80.98%.

**Step 6: Write your Algorithm**
An algorithm was written that accepts a file path to an image and first determines whether the image contains a human, dog, or neither. If a dog is detected in the image, it returns the predicted breed. If a human is detected in the image, it returns the resembling dog breed. If neither is detected in the image, an error message is displayed.
The face_detector and dog_detector functions developed in steps 1 and 2 above were used. The CNN from Step 5 was used to predict the dog breed.

**Step 7: Test Your Algorithm**
The algorithm was tested on three dog images, three human images and an image of a cat. It predicted the three dog breeds accurately, and did not mistaken the cat image for a dog. It chose the breed a human resembles as well. The output was better than anticipated and is shown below.  

![image](https://github.com/nirvannar/Capstone-Dog-Breed/assets/52913504/b3d184b0-7872-45b1-a440-c75e57fb1d0d)

![image](https://github.com/nirvannar/Capstone-Dog-Breed/assets/52913504/028a7f9b-23c0-43ea-8c20-f1e79ff2876f)

![image](https://github.com/nirvannar/Capstone-Dog-Breed/assets/52913504/08ad2cd7-1257-42dd-b799-4721242c8c42)

![image](https://github.com/nirvannar/Capstone-Dog-Breed/assets/52913504/c62c7767-d6a0-4b3e-a83d-0482f634157b)

![image](https://github.com/nirvannar/Capstone-Dog-Breed/assets/52913504/92720855-b65f-4ab7-85d9-07720308d95b)

![image](https://github.com/nirvannar/Capstone-Dog-Breed/assets/52913504/e80162e9-d2b7-4b38-964f-e4cb7b0d21e2)

![image](https://github.com/nirvannar/Capstone-Dog-Breed/assets/52913504/954586f8-8c3c-4609-bee6-4e299ff68f45)


[Medium blog](https://medium.com/@nirvannsramp/dog-breed-classifier-b485852f6b0d)


## Licensing, Authors, and Acknowledgements<a name="Licensing,-Authors,- and- Acknowledgements"></a>

[Udacity](https://learn.udacity.com/nanodegrees/nd025/parts/cd1971/lessons/e3f2b25a-5e8a-404f-a1f8-18c3524ce867/concepts/98fa9de4-2ae3-4e3c-9fda-9bdbde2bf459)

[Github](https://github.com/)

[Labrador Retriever image](https://hips.hearstapps.com/hmg-prod/images/dog-1598970334.png?crop=0.563xw:1.00xh;0.231xw,0&resize=980:*)

[Chow chow dogs image](https://hips.hearstapps.com/hmg-prod/images/chow-chow-dog-breed-651d88c291c70.jpg?crop=0.668xw:1.00xh;0.197xw,0&resize=1200:*)

[Portrait of a surprised cat Scottish Straight](https://stock.adobe.com/search?k=cat&asset_id=97589769)  

[Algoscale](https://algoscale.com/blog/yolo-vs-ssd-which-one-is-a-superior-algorithm/#:~:text=The%20main%20difference%20between%20YOLO,score%20of%20more%20than%200.5.)

[dlib.com](http://dlib.net/imaging.html)

[techtarget.com](https://www.techtarget.com/searchenterpriseai/definition/face-detection) 

[faceonlive.com](https://faceonlive.com/face-detection-models-the-ultimate-guide-2023-unleash-the-power-of-ai-to-spot-faces-like-a-pro/#:~:text=Dlib's%20CNN%3A%20Ideal%20for%20applications,recognition%20systems%20or%20video%20surveillance.)

[Intel](https://www.intel.com/content/www/us/en/internet-of-things/computer-vision/convolutional-neural-networks.html#:~:text=A%20CNN%20model%20is%20a,training%2C%20optimizing%2C%20and%20inference.)

[GeeksforGeeks](https://www.geeksforgeeks.org/ml-machine-learning/)

[ResearchGate](https://www.researchgate.net/figure/Relationship-between-AI-ML-DL-and-CNN_fig1_358745776)

[CNN transfer learning](https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce)

[Ai, ML and DL differences](https://towardsdatascience.com/understanding-the-difference-between-ai-ml-and-dl-cceb63252a6c)

[artificial intelligence](https://www.britannica.com/technology/artificial-intelligence) 

[Exploring ResNet50: An In-Depth Look at the Model Architecture and Code Implementation](https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f#:~:text=ResNet50%20is%20a%20deep%20convolutional,which%20is%2050%20layers%20deep.)

[GAP](https://medium.com/r/?url=https%3A%2F%2Fpaperswithcode.com%2Fmethod%2Fglobal-average-pooling)

[Global Average Pooling](https://medium.com/r/?url=https%3A%2F%2Fiq.opengenus.org%2Fglobal-average-pooling%2F)








