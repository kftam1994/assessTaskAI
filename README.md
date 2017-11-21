# Assessment Task for AI/Digital Analyst

This is a simple HTML page taking an image with English words and numbers and returning the text on it. An English word image generator produce character images in different font and font sizes. Then, those images generated and other character images are fed to a neural network model. The idea of the preprocessing method and model is from [here](http://francescopochetti.com/text-recognition-natural-scenes/#first). The model is used to retrieved the text from the image.

![index_screen](https://user-images.githubusercontent.com/33834357/33052612-a88e3248-ceaa-11e7-9a9f-095aaefc50e0.JPG)

The result page:

![result_screen](https://user-images.githubusercontent.com/33834357/33053394-4c428b3e-ceae-11e7-8717-8dc2e414cd51.JPG)

## Getting Started

To run the page, it needs Python 3 to be installed. It also needs packages including
 * Matplotlib
 * scikit-image(skimage)
 * scikit-learn(sklearn)
 * Flask
Packages can be installed through pip.

The HTML page can run when Flask_web.py is executed with Python 3 in the command line through the command "python Flask_web.py"

"Running on http://127.0.0.1:5000/" will be displayed.

Then, the page can be browsed at the address http://127.0.0.1:5000

An image with English words and numbers on it can be uploaded. The text will be retrieved and displayed in the result image after processing.

## English Word Generator

An English word generator produces images of English characters and numbers for each font and each font size in a black background. It includes 26 upper case and lower case characters, and numbers, 0 to 9. Those 53 font files are in "src_smallSubset/Font/" folder. 4 Font sizes inlcuding, 8,12,36,40, are used. 13144 images are produced and saved to the folder "words_generated".

## Other Dataset

Apart from English Word Generator, other English character and number datasets are also inputted to improve the performance of the model. Datasets of different variation of character include computer font character from [Chars74k dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/), a subset of street view character image from [dataset](https://www.kaggle.com/c/street-view-getting-started-with-julia/data), and a handwritten character dataset from [EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset).
Moreover, an object image dataset from [CIFAR-10](https://www.kaggle.com/c/cifar-10/data) is used to recognize objects other than character and number.

## Preprocessing

Street view character image, handwritten character and object image datasets are noisy images from photos in Google Street View or scanning of handwritten document. 

## Support Vector Machine Classifier

## Neural Netwrok Model

## HTML page

## Contact 
If you have any questions or suggestions feel free to contact me at <kftam@connect.ust.hk>.
