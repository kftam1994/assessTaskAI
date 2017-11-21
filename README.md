# Assessment Task for AI/Digital Analyst (Andy Tam)

This is a HTML page taking an image with English words and numbers and returning the text on it. An English word image generator produce character images in different font and font sizes. Then, those images generated and other character images are fed to a neural network model. The idea of the preprocessing method and model is from [here](http://francescopochetti.com/text-recognition-natural-scenes/#first). The model is used to retrieved the text from the image.

![index_screen](https://user-images.githubusercontent.com/33834357/33052612-a88e3248-ceaa-11e7-9a9f-095aaefc50e0.JPG)

The result page:

![result_screen](https://user-images.githubusercontent.com/33834357/33080444-1247f2c4-cf13-11e7-8cff-b9827331e679.JPG)

## Getting Started

To run the page, it needs Python 3 to be installed. It also needs packages including
 * Matplotlib
 * Scikit-image([skimage](http://scikit-image.org/))
 * Scikit-learn([sklearn](http://scikit-learn.org/stable/))
 * [Flask](http://flask.pocoo.org/)

Packages can be installed through pip.

The HTML page can run when Flask_web.py is executed with Python 3 in the command line through the command 
```
python Flask_web.py
```

"Running on http://127.0.0.1:5000/" will be displayed.

Then, the page can be browsed at the address http://127.0.0.1:5000

An image with English words and numbers on it can be uploaded. The text will be retrieved and displayed in the result image after processing.

## Architecture

Flow diagram

![flow_diagram_assesstaskai](https://user-images.githubusercontent.com/33834357/33081420-49d5b9f4-cf15-11e7-81cf-f61bb0b20025.png)

### English Word Generator

An English word generator produces images of English characters and numbers for each font and each font size in a black background. It includes 26 upper case and lower case characters, and numbers, 0 to 9. Those 53 font files are in "src_smallSubset/Font/" folder. 4 Font sizes, 8,12,36,40, are used. 13144 images are produced and saved to the folder "words_generated".

### Other Dataset

Apart from English Word Generator, other English character and number datasets are also inputted to improve the performance of the model. Datasets of different variation of character include:

 * 6979 computer font character images from [Chars74k](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) dataset
 * A subset of 6283 street view character image from Chars74k dataset in [here](https://www.kaggle.com/c/street-view-getting-started-with-julia/data)
 * A handwritten character dataset of 14000 images from [EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset)
 
Moreover, an non-text object image dataset from [CIFAR-10](https://www.kaggle.com/c/cifar-10/data) is used to recognize non-text objects other than character and number.

### Preprocessing and Extracting Features

Images are preprocessed and then features are extracted before feeding them to the models. Street view character image, handwritten character and non-text object image datasets are noisy images from photos in Google Street View or scanning of handwritten document. Scikit-image package is used to preprocess and extract features. 

Images are changed to [grayscale](http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2gray) and then applied [median filter](http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median) to reduce noise. [Rescaling intensity](http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity) is also applied to the images to stretch contrast. After [resizing](http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize) to 32x32 pixels, the feature vectors of images are retrieved through [histogram of oriented gradient](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html), which computes gradient image and gradient histograms.

Computer font character and words from English Generator are also changed to grayscale and resized to 32x32 pixels before extracting features through histogram of oriented gradient.

The lists of features for each image are saved to a txt file.

### Support Vector Machine Classifier

Support Vector Machine Classifier (SVC) aims at identifying and differentiating characters from other non-text object. Scikit-learn [SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) package is used to build the model. Radial basis function is used as the kernel function. 

The following features are loaded, shuffled randomly and inputted to train the SVC model.

 * Street view character and 
 * Non-text object images  

The average recall of model through cross validation is 89%. The model is then saved to svmmodel.pkl file after fitting data.

### Neural Netwrok Model

Multi-layer Perceptron (MLP) neural network model aims at classifying an image to 62 classes of upper case and lower case characters, and digits by backpropagation. Scikit-learn [MLP](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) package is used to build the model. The following are the parameters:

 * Solver: stochastic gradient-based optimizer
 * Activation function: hyperbolic tan function
 * Learning rate, "invscaling": gradually decreases the learning rate
 * 5 hidden layers with 300 neurons per layer, and one input and output layer

The following features are combined, shuffled randomly and inputted to train the MLP model. 

 * Street view character image, 
 * Handwritten character image, 
 * Computer font character image and 
 * Image generated by English Word Generator 
 
The average accuracy of model through cross validation is 68%. The model is then saved to mlpfullmodel.pkl file after fitting data.

### HTML page

The HTML page is a simple interface for user to upload an image, process by predicting the text and display the output image. It is built with [Flask](http://flask.pocoo.org/), which is a tool to create a page and run python function in the HTML page. 

The image is firstly uploaded by the user in the index.html page and saved in the "static" folder. The image is then read by the function to predict text in the image, and print and save the text on a blank background. The function look for objects in the image through searching for contours at level 0.45 by [find.contours](http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours) of Scikit-image package. 

After finding contours, SVC model is loaded and takes the cropping images of objects appearing in the image to filter out images containing non-text objects. MLP model is then loaded and takes the filtered images to identify each character and number. The text is printed on a blank background image, and this result image is saved to the "static" folder. The original image and the result image are displyed in the result.html page.

Images in "src_smallSubset/examples_for_test/" folder are example images to be uploaded to the page for demonstrating the result.

## Improvement

 * [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)
 
Convolutional neural network is a deep, feed-forward artificial neural networks for image processing and recognition. It includes multiple Convolutional and Pooling layers to learn the image in pixels. It has an advantage of less preprocessing procedures required when it is compared to my model.
 
 * A full [Chars74k](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) dataset
 
The full Chars74k dataset contains 50000 character images from street view, computer font and handwritten text. Using the whole dataset provides a more comprehensive examples of characters in different sizes, fonts and contexts, which may benefit the machine learning model.

## Contact 
If you have any enquiries or suggestions, please feel free to contact me at <kftam@connect.ust.hk>.
