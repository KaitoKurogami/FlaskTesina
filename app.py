from flask import Flask, render_template, request,redirect,flash,session
from werkzeug.utils import secure_filename
import os
from forms.index_form import FullForm
import time

app = Flask(__name__)

app.config['SECRET_KEY'] = "1234" #since it's local, i dont care for security, i only have it because it's needed
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['RESULT_FOLDER'] = 'static/files/results'


@app.route('/',methods=('GET','POST'))
def index():
    form = FullForm()
    if request.method=='POST':
        print(request.form)
        print(form.validate_on_submit())
        if form.validate_on_submit():
            print("*********************")
            print(request)
            print("+++++++++++++++++++++")
            print(request.values)
            print("---------------------")
            print(request.form)
            print("/////////////////////")
            file = request.files["file-file"] #grab the file
            information = preprocesor(request.form,file.filename)
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) #save the file
            #return "file has been uploaded."
            time.sleep(3)
            flash("Procesamiento completado, imagenes guardadas en la carpeta "+app.config['RESULT_FOLDER'])
            return redirect("/",code=302)
        
    return render_template('index.html',form=form)

def preprocesor(multiDict,filename):
    upload_adress=os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],secure_filename(filename))
    nets = multiDict.getlist('nets')    #list of CNNs to use
    keys = multiDict.keys() #the keys to later check for the checkboxes
    config = {} #start the dictionary with all the info for the program
    config["models"]=nets
    config["visualizers"]=[]
    config["filename"]=filename
    for key in keys:    #iterate the keys
        match key:  #search for the checkboxes and configure each visualization model accordingly
            case 'check-SHAP':
                shap={}
                shap["evals"]=multiDict.get("shap-SHAP_evals")
                shap["batch_size"]=multiDict.get("shap-SHAP_batch_size")
                config["visualizers"].append({"shap":shap})
            case 'check-LIME':
                lime={}
                lime["perturbations"]=multiDict.get("lime-LIME_perturbations")
                lime["kernel_size"]=multiDict.get("lime-LIME_kernel_size")
                lime["max_dist"]=multiDict.get("lime-LIME_max_dist")
                lime["ratio"]=multiDict.get("lime-LIME_ratio")
                config["visualizers"].append({"lime":lime})
            case 'check-GradCAM':
                gradCAM={}
                config["visualizers"].append({"gradCAM":gradCAM})
    return config

if __name__ == "__main__":
    app.run(debug=True)