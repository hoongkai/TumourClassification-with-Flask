from flask import Flask, render_template, redirect, url_for, request
from keras.models import load_model
from keras_preprocessing import image
import numpy as np
from PIL import Image
import cv2
from classifier import check
import os

app = Flask(__name__)

dic = {0 : 'No Tumour Detected', 1 : 'Tumour Is Found'}

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path)
	i_array = np.asarray(i)
	#x = np.resize(i_array,(1,240,240,3))
	x = np.resize(i_array,(1,200,200,1))
	p = model.predict(x)
	p_class=np.argmax(p,axis=1)
	print(p_class)
	print("hehe")
	return dic[p_class[0]]



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("app.html")

@app.route("/upload", methods=['GET','POST'])
def upload():
	return render_template("prediction.html")

@app.route("/prediction", methods = ['POST'])
def prediction():
	if request.method == 'POST':
		img = request.files["my_image"]

		img_path = "static/" + img.filename	
		img.save(img_path)

		#i = image.load_img(img_path)
		#print(type(i))
		#i_array = np.asarray(i)
		#print(type(i_array))
		#x = np.resize(i_array,(200,200,1))
		#print(type(x))
		#print(x.shape)

		p = predict_label(img_path)
		
	return render_template("prediction.html",prediction = p, img_path =img_path)
	
if __name__ =='__main__':
#app.debug = True
	app.run(host='0.0.0.0', port=80, debug = True)