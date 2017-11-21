# Assessment Task for AI/Digital Analyst

This is a simple HTML page taking an image with English words and numbers and returning the text on it. An English word image generators produce character images. Then, those images generated and other character images are fed to a neural network model. The model is used to retrieved the text from the image.
![index_screen](https://user-images.githubusercontent.com/33834357/33052612-a88e3248-ceaa-11e7-9a9f-095aaefc50e0.JPG)

The result page

![result_screen](https://user-images.githubusercontent.com/33834357/33052636-ca447d0c-ceaa-11e7-9b91-d71914a0c993.JPG)

## Getting Started

To run the page, it needs Python 3 to be installed. It also needs packages including
 * Matplotlib
 * scikit-image(skimage)
 * scikit-learn(sklearn)
Packages can be installed through pip.

The HTML page can run when Flask_web.py is executed with Python 3 in the command line through the command "python Flask_web.py"

"Running on http://127.0.0.1:5000/" will be displayed.

Then, the page can be browsed at the address http://127.0.0.1:5000

An image with English words and numbers on it can be uploaded. The text will be retrieved and displayed in the result image after processing.
