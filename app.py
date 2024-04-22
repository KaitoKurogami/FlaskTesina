from flask import Flask, render_template, request,redirect,flash,session,url_for
from werkzeug.utils import secure_filename
import os
from forms.index_form import FullForm
import time
from core import core

try:
   os.makedirs("./static/files")
except FileExistsError:
   # directory already exists
   pass


try:
   os.makedirs("./static/files/results")
except FileExistsError:
   # directory already exists
   pass

try:
   os.makedirs("./static/files/models")
except FileExistsError:
   # directory already exists
   pass

#for some reason, conda couldn find the SSL certificate in a new VM, if you dont have this problem, just comment the next to lines to feel safer
import ssl
ssl.create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

app = Flask(__name__)

app.config['SECRET_KEY'] = "cbfbd8c99f134afa075d13b655662ed09d3fa29c46a09cd55ad37e3380a93a0e" #since it's local, i dont care for security, i only have it because it's needed
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['RESULT_FOLDER'] = 'static/files/results'
app.config['vgg16']= tf.keras.applications.vgg16.VGG16(weights="imagenet")
app.config['resnet50']= tf.keras.applications.resnet50.ResNet50(weights="imagenet")

@app.route('/',methods=('GET','POST'))
def index():
    form = FullForm()
    if request.method=='POST':
        if form.validate_on_submit():
            file = request.files["file-file"] #grab the file
            if "Red propia" in request.form.getlist('nets'):
                app.config['classes']=request.form.get("classesText-classesText").split(",")
                app.config['Red propia']= tf.keras.models.load_model(app.config['UPLOAD_FOLDER']+"/models/"+request.files["newNet-newNet"].filename,compile=False)
            configurationCore = preprocesor(request.form,file.filename) #create te dictionary with the configuration for the app
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) #save the file
            filenames=core(configurationCore)
            session['filenames']=filenames
            return redirect('/results')
        
    return render_template('index.html',form=form)

@app.route('/results')
def results():
    images=session['filenames']
    session.pop('filenames',default=None)
    return render_template('results.html',images=images)

def preprocesor(multiDict,filename):
    nets = multiDict.getlist('nets')    #list of CNNs to use
    keys = multiDict.keys() #the keys to later check for the checkboxes
    config = {} #start the dictionary with all the info for the program
    config["models"]=nets
    config["visualizers"]={}
    config["filename"]=filename
    for key in keys:    #iterate the keys
        match key:  #search for the checkboxes and configure each visualization model accordingly
            case 'check-SHAP':
                shap={}
                shap["evals"]=multiDict.get("shap-SHAP_evals")
                shap["batch_size"]=multiDict.get("shap-SHAP_batch_size")
                config["visualizers"]["shap"]=shap
            case 'check-LIME':
                lime={}
                lime["perturbations"]=multiDict.get("lime-LIME_perturbations")
                lime["kernel_size"]=multiDict.get("lime-LIME_kernel_size")
                lime["max_dist"]=multiDict.get("lime-LIME_max_dist")
                lime["ratio"]=multiDict.get("lime-LIME_ratio")
                config["visualizers"]["lime"]=lime
            case 'check-GradCAM':
                gradCAM={}
                config["visualizers"]["gradCAM"]=gradCAM
    return config

if __name__ == "__main__":
    app.run(debug=True)