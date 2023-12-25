# Capstone-Dog-Breed
Udacity Capstone Project

If you submit a blog post, you will need to submit two links - one for your GitHub repo and one for your blog post. The audience for your blog post should be that of a technical audience member. Your GitHub repo should contain your code (if possible), dataset (unless you chose the Spark project option), and README.md files.

# Table of Contents

* [Installation](#Installation)
* [Project Motivation](#Project-Motivation)
* [File Descriptions and Analyses](#File-Descriptions-and-Analyses)
* [Results](#Results)
* [Licensing, Authors, and Acknowledgements](#Licensing,-Authors,-and-Acknowledgements)


## Installation <a name="Installation"></a>
The code should run with no issues using Python versions 3.8.8 using Jupyter Notebook server version 6.3.0.  Numpy, glob, keras.utils, sklearn.datasets, random, cv2, matplotlib.pyplot, keras.applications.resnet50, keras.preprocessing, tqdm, PIL, keras.layers, keras.models, keras.optimizers, keras.callbacks, and pickle were the libraries used.  

## Project Motivation <a name="Project-Motivation"></a>
For this project given an image of a dog, the classifier housed in a Jupyter notebook is expected to identify the breed. If supplied an image of a human, the code will identify the resembling dog breed. Otherwise it will output an error message. 



**_Business Problem Understanding_**
The project will be divided into the following tasks

**I. Exploratory Data Analysis**
The data was explored with descriptive statistics.

**II. Rank Based Recommendations**
To get started in building recommendations, I first found the most popular articles based on the most interactions. Since there are no ratings for any of the articles, the assumption was that the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).

**III. User-User Based Collaborative Filtering**
In order to build better recommendations for the users of IBM's platform, I looked at users that are similar in terms of the items they have interacted with. These items could then be recommended to similar users. This results in more personal recommendations for the users.

**IV. Matrix Factorization**
Finally a machine learning approach to building recommendations was used. Using the user-item interactions,  a matrix decomposition was built. 



## File Descriptions and Analyses <a name="File-Descriptions-and-Analyses"></a>
**_Data Preparation_**  
'data/user-item-interactions.csv' and 'data/articles_community.csv' were explored.

## Results <a name="Results"></a>

**I. Exploratory Data Analysis**
The median_val, user_article_interactions, max_views_by_user, max_views, most_viewed_article_id, unique_articles, unique_users,total_articles were explored.

**II. Rank Based Recommendations**
Rank Based recommendations for new users is the best way of making recommendations for this dataset.  The most popular articles based on the most interactions used two functions which pulled the top ids and the top names. The functions get_top_articles and get_top_articles_ids from Part II could be used to get the top articles to recommend to a new user. This problem described above is known as the cold start problem.

**III. User-User Based Collaborative Filtering**
A user-item matrix was created.  In order to build better recommendations for the users of IBM's platform, I looked at users that are similar in terms of the items they have interacted with. These items could then be recommended to similar users. This results in more personal recommendations for the users. It will not be possible to use User-User Based Collaborative Filtering Recommendations as information (such as ratings) on user-article interactions for a new user as ratings are not known, and this method is based on using the collaboration of user-item interactions.

Content-based similarity method for recommendations could be used for a new user. In this method, recommendations are based using information about the content of the articles and ranking of the highest ranked articles based on specific content or associated with some term. Natural Language Processing (NLP) can be used in this method. Content could be from the columns doc_body, doc_description, or doc_full_name from the 'data/articles_community.csv' file.

**IV. Matrix Factorization**
Finally a machine learning approach to building recommendations was used. Using the user-item interactions,  a matrix decomposition was built. Singular Value Decomposition from Numpy was performed.  

The SVD in the lesson did not converge as it had encountered a NaN or missing values in the user_movie_subset matrix. There was a significant missing data. "The number of ratings made for user-movie pairs that didn't have ratings is 13 835 713". This means it was a very sparse matrix.  SVD from NumPy does not work when the matrices do not have a value in every cell and is thus considered not complete. FunkSVD which uses a stochastic gradient descent formula was used in the lesson to address NaN/missing values.  SVD in this Project is used on the user_item_matrix which has no missing values.

The training data suggests that as latent features increase the accuracy increases. However the test data suggests that accuracy decreased as latent features increased. There is direct conflict between the two. This difference could be due to the small overlap of users between training and testing datasets. Only 20 users in the test dataset are shared with the training dataset and were used to give the result.

We could increase the sizes of the test datasets, use online recommendation evaluations such as A/B testing, or time-based testing as opposed to matrix factorization used above which is an offline method.

All types of recommendations should be used not just user-user based collaborative and rank-based recommendations. Content-based like that desctibed in Part III question (6) for new users and knowledge-based recommendations for existing could be used.


[Medium blog](https://medium.com/@nirvannsramp/dog-breed-classifier-b485852f6b0d)

