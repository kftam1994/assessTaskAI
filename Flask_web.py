#Flask is used to provide a simple HTML interface for uploading an image and displaying the result image after processing
#HTML files are put in the "template" folder
#Images are saved and accessed in the "static" folder
#Package Flask required
import os
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename

#reference: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if allowed_file(file.filename)==False:
            flash('Not an image')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            text=process(filename)
            return render_template('result.html',
				   image_value=os.path.join(app.config['UPLOAD_FOLDER'], filename),
				   result_value=os.path.join(app.config['UPLOAD_FOLDER'], "result_"+filename),
				   result_text=text)
        
    return render_template('index.html')


def process(filename):
	#A function to predict text in the image, and print and save the text on a blank background
	#Packages matplotlib, skimage and sklearn required
	import numpy as np
	import matplotlib.pyplot as plt
	from skimage.io import imread,imshow
	from skimage import measure
	from skimage import color
	from skimage.transform import resize
	from skimage.feature import hog
	from sklearn.externals import joblib

	#Change image from RGB color image to a grayscale image
	image=color.rgb2gray(imread(os.path.join(app.config['UPLOAD_FOLDER'], filename)))

	#Look for objects in the image through contour finding
	#reference: http://scikit-image.org/docs/dev/auto_examples/edges/plot_contours.html
	contours = measure.find_contours(image,0.45) #0.45
	#Extract features through Histogram of Oriented Gradient and create a list of features
	features_test=[]
	list_image=[]
	for n,contour in enumerate(contours):
		max_r=int(round(np.amax(contour[:,0])))
		min_r=int(round(np.amin(contour[:,0])))
		max_c=int(round(np.amax(contour[:,1])))
		min_c=int(round(np.amin(contour[:,1])))
		if(max_c-min_c>10 and max_c-min_c<50 and max_r-min_r>10 and max_r-min_r<50):
			list_image.append([min_r,max_r,min_c,max_c])
			imageResized = resize( image[min_r:max_r,min_c:max_c], (32,32) )
			features_extraction, hog_image = hog(imageResized, cells_per_block=(1, 1), visualise=True)
			features_test.append(features_extraction.tolist())

	#Load fitted and saved Support Vector Machine Classifier (SVC)
	svmclf = joblib.load('svmmodel.pkl') 
	#Differentiate and identify the character from other objects in the image by SVC model
	filter_char = svmclf.predict(features_test)
	features_test_extracted=np.array(features_test)[filter_char.astype(bool)]
	list_image_extracted=np.array(list_image)[filter_char.astype(bool)]
	#Load fitted and saved Multi-layer Perceptron (MLP) neural network model
	mlpclf_full = joblib.load('mlpfullmodel.pkl') 
	#Retrieve the text by identifying each character by MLP model
	result = mlpclf_full.predict(features_test_extracted)

	#Show the text in a blank background image
	fig, ax = plt.subplots(figsize=(15,15))
	blank_image=np.zeros(image.shape,dtype=np.uint8)
	ax.imshow(blank_image, interpolation='nearest', cmap=plt.cm.gray)
	for point,text in zip(np.array(list_image)[filter_char.astype(bool)].tolist(),result):
		ax.text((point[3]+point[2])/2,(point[1]+point[0])/2,text,color="white",fontsize=25)
	ax.axis("image")
	ax.set_xticks([])
	ax.set_yticks([])
	try:
    		os.remove(os.path.join(app.config['UPLOAD_FOLDER'], "result_"+filename))
	except OSError:
    		pass
	#Save the result
	plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], "result_"+filename))
	plt.close("all")
	return result
			   
							   
#Run the web app							   
if __name__ == '__main__':
   app.run()