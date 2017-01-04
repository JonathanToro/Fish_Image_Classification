# Catching Fish With Neural Nets
## Goal: Use computer vision and CNNs to accurately detect fish species in an image

My main approach to tackling this Kaggle competition was by using Convolutional Neural Networks. I used two different pretrained neural networks and architectures provided by Google. My best results was by using the InceptionV3 model and augmenting the test data so that the model had multiple tries to classify each image correctly. I hosted the modeling on Amazon Web Services by using their g2.8x instance which provided me with 4 gpus to work with. The code to parallelize the training is provided above. At the time of the submission of my best model it ranked top 5% on Kaggle.

Please refer to my [blog][1] for more information.

[1]: https://jonathantoro.github.io/Catching-Fish-With-Neural-Nets/
