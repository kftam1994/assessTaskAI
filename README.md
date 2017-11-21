# Assessment Task for AI/Digital Analyst

This is a simple HTML page taking an image with English words and numbers and returning the text on it. An English word image generator produce character images in different font and font sizes. Then, those images generated and other character images are fed to a neural network model. The idea of the preprocessing method and model is from [here](http://francescopochetti.com/text-recognition-natural-scenes/#first). The model is used to retrieved the text from the image.

![index_screen](https://user-images.githubusercontent.com/33834357/33052612-a88e3248-ceaa-11e7-9a9f-095aaefc50e0.JPG)

The result page:

![result_screen](https://user-images.githubusercontent.com/33834357/33053394-4c428b3e-ceae-11e7-8717-8dc2e414cd51.JPG)

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

## Program Structure

## English Word Generator

An English word generator produces images of English characters and numbers for each font and each font size in a black background. It includes 26 upper case and lower case characters, and numbers, 0 to 9. Those 53 font files are in "src_smallSubset/Font/" folder. 4 Font sizes, 8,12,36,40, are used. 13144 images are produced and saved to the folder "words_generated".

## Other Dataset

Apart from English Word Generator, other English character and number datasets are also inputted to improve the performance of the model. Datasets of different variation of character include 6979 computer font character images from [Chars74k dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/), a subset of 6283 street view character image from [dataset](https://www.kaggle.com/c/street-view-getting-started-with-julia/data), and a handwritten character dataset of 14000 images from [EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset).
Moreover, an object image dataset from [CIFAR-10](https://www.kaggle.com/c/cifar-10/data) is used to recognize objects other than character and number.

## Preprocessing and Extracting Features

Images are preprocessed and then features are extracted before feeding them to the models. Street view character image, handwritten character and object image datasets are noisy images from photos in Google Street View or scanning of handwritten document. Scikit-image package is used to preprocess and extract features. Images are changed to grayscale and then applied median filter to reduce noise. Rescaling intensity is also applied to the images to stretch contrast. After resizing to 32x32 pixels, the feature vectors of images are retrieved through histogram of oriented gradient, which computes gradient image and gradient histograms.
Computer font character and words from English Generator are also changed to grayscale and resized to 32x32 pixels before extracting features through histogram of oriented gradient.
The lists of features for each image are saved to a txt file.

## Support Vector Machine Classifier

Support Vector Machine Classifier (SVC) aims at identifying and differentiating characters from other non-text object. Scikit-learn [SVM](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) package is used to build the model. Radial basis function is used as the kernel function. Features of street view character and object images are loaded, shuffled randomly and inputted to train the SVC model. The average recall of model through cross validation is 89%. The model is then saved to svmmodel.pkl file after fitting data.

## Neural Netwrok Model

Multi-layer Perceptron (MLP) neural network model aims at classifying an image to 62 classes of upper case and lower case characters, and digits. Scikit-learn [MLP](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) package is used to build the model. The following are the parameters:

 * Solver: stochastic gradient-based optimizer
 * Activation function: hyperbolic tan function
 * Learning rate, "invscaling": gradually decreases the learning rate
 * 5 Hidden Layers with 300 neurons

Features of street view character image, handwritten character image and image generated by English Word Generator are combined, shuffled randomly and inputted to train the MLP model. The average accuracy of model through cross validation is 68%. The model is then saved to mlpfullmodel.pkl file after fitting data.

## HTML page

The HTML page is a simple interface for user to upload an image, process by predicting the text and display the output image. It is built with [Flask](http://flask.pocoo.org/), which is a tool to create a page and run python function in the HTML page. The image is firstly uploaded by the user and saved in the "static" folder. The image is then read by the function to predict text in the image, and print and save the text on a blank background. The function look for objects in the image through searching for contours at level 0.45 by [find.contours](http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours) of Scikit-image package. After finding contours, the cropping image of objects appearing in the image 

## Improvement

CNN
Full Chars74k dataset

## Contact 
If you have any questions or suggestions feel free to contact me at <kftam@connect.ust.hk>.
