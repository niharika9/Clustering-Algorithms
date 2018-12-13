# Clustering-Algorithms

## Objectives 
- Problem 1 : To Implement KMeans Clustering Algorithm without the use of inbuilt libraries in Spark Scala.
- Problem 2 : To Implement  KMeans Clustering and Bisecting KMeans Clustering Algorithm using Spark Mllib API in Scala

## Environment Requirements
- Scala 2.11 version
- Spark 2.3.1 version


## Dataset
Yelp Challenge Dataset. The dataset was prepared by extracting the text from reviews in Yelp Challenge Dataset Round 12. 
All special characters were removed from the reviews by running ​sub(​​'\W+'​​, ​​''​​,word) ​and then the words were converted to lower case.
Only reviews with more than 50 words were selected and one review is written per line. The small dataset (yelp_reviews_clustering_small) contains 1000 reviews. 
The large dataset (yelp_reviews_clustering_large) contains 100K reviews.

## Problem 1 
(KMeans from Scratch), for this , I have used 2 features : Word Count and TF-IDF . For Word Count feature , I have used the TF(Token Frequency ) API which uses a vector that counts its occurences in the document. For TF- IDF, I have used the spark mllib API which converts a document to a vector with each string having its tf-idf value which is the product of token frequency and inverse document frequency
After building the features for the input, I have implemented the k-means algorithm

For this, I have initially picked up “N” random input vectors and stored them as
centroids , Now in a loop of “iterations” , I have used the following logic
```
1. Set the points to the centroid it is closest to (I have used Euclidean distance as
closeness measure )
2. Now, I recalculated the centroids for each cluster, using the average of all points
in the cluster as its centroid
3. Repeated step 1 and 2 until loop iterations condition is satisfied
```

## Problem 2
(Inbuilt Library Code)
For this, I have used k-means and Bisecting K means Spark Mllib API . I have used TF IDF api for generating input feature vector.
```
1. First trained a model using the input features which generates a metadata to predict which cluster any vector belongs. 
2. This model only predicts and doesn’t store all the points in the cluster
3. So, using the trained model, I predicted which cluster each input vector belongs to and stored them in a map to be able to use for calculating statistics.
```

## Command to Run the code 
For Problem1 :
I have used Word Count ”W” and TF-IDF “T” feature vectors for input
```
spark-submit --class Niharika_Gajam_Task1 Clustering.jar <input_file.txt> <feature> <num_of_clusters> <iterations>
```
For Problem2 :
I have used kmeans “k” and bisecting kmeans “B” algorithm
``` 
spark-submit --driver-memory 6G --class Niharika_Gajam_Task2 Clustering.jar <input_file.txt> <algorithm> <num_of_clusters> <iterations>
```
NOTE: for bisecting k means , the code throws memory out of bounds error. So adding “—driver-memory 6G” will make it work .



