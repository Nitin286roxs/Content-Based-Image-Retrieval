# Content-Based-Image-Retrieval
 Content-based image retrieval, a technique that uses visual contents to search images from large scale image databases. 

# Procedure:
![alt text](https://raw.githubusercontent.com/Nitin286roxs/Content-Based-Image-Retrieval/master/procedue.png)
 
# Used model for feature extraction:
In this assignment, I have used keras for feature extraction. There are following
models available in keras 
![alt text](https://raw.githubusercontent.com/Nitin286roxs/Content-Based-Image-Retrieval/master/keras_model.png)


A number of comparisons can be drawn:
- AlexNet and ResNet-152, both have about 60M parameters but there is about 10% difference in their top-5 accuracy. But training a ResNet-152 requires a lot of computations (about 10 times more than that of AlexNet) which means more training time and energy required.
- VGGNet not only has a higher number of parameters and FLOP as compared to ResNet-152 but also has a decreased accuracy. It takes more time to train a VGGNet with reduced accuracy.
- Training an AlexNet takes about the same time as training Inception. The memory requirements are 10 times less with improved accuracy (about 9%)

I have used ResNet50 which is smaller than ResNet152 and less training time and less energy required and accuracy is very close to ResNet152.

# Requirements to run this project:

    • keras
    • tensorflow>=2.2
    • numpy
    • matplotlib

# Algorithm:
Basically I am storing features of all images in feats.npy during OFFLINE Phase. I have extracted feature for query image. 

For similarity measure, I am doing numpy dot product loaded transposed ndarry [4778*2048] feats.np and query image ndarry [2048,1]. After dot product we are getting [4778,1]. Then I had sorted this ndarry in decending order of score. 

I am taking only top 9 similar images on the basis of score.

LOGIC:
```sh
scores = np.dot(query_vec, self.feats.T)
rank_id = np.argsort(scores)[::-1]
rank_score = scores[rank_id]
maxres = 9
return [self.imgnames[index] for i, index in enumerate(rank_id[0:maxres])]
```

# Run the project:
TO run the project follow the command given below:
```sh
optional arguments:
  -dataset DATASET      Path to dataset
  -img_query IMG_QUERY  image to be search
```
Syntax:
```sh

python3 run_cbir.py -dataset <PATH TO DATASET> -img_query <IMAGE TO BE SEARCH>
```
Example:
```sh
python3 run_cbir.py -dataset ./dataset -img_query ./dataset/56.jpg
```
# Sample Output
![alt text](https://raw.githubusercontent.com/Nitin286roxs/Content-Based-Image-Retrieval/master/result1.png)
![alt text](https://raw.githubusercontent.com/Nitin286roxs/Content-Based-Image-Retrieval/master/result2.png)


