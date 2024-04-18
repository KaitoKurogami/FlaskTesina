from tensorflow.keras.applications.resnet50 import preprocess_input as ResNet50preprocess_input, decode_predictions as ResNet50decode_predictions
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16preprocess_input, decode_predictions as vgg16decode_predictions
from flask import current_app as app
from gradCAM import GRADCAM
from LIME import LIME
from SHAP import SHAP
from runNet import runNet
import os
from werkzeug.utils import secure_filename
from datetime import datetime

def core(config):
    now = datetime.now()
    #make the image address to tell the CNNs where to find it
    filenames=[]
    image_address=os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],secure_filename(config["filename"]))
    CNNs={}
    for model in config["models"]: #iterate for each CNN model and make the configuration required
        match model:
            case "VGG16":
                CNNs["VGG16"]={}
                CNNs["VGG16"]["model"]=app.config["vgg16"]
                CNNs["VGG16"]["preprocess_input"]=vgg16preprocess_input
                CNNs["VGG16"]["decode_predictions"]=vgg16decode_predictions
                CNNs["VGG16"]["last_conv_layer_name"]="block5_conv3"
                CNNs["VGG16"]["scale"]=224 / 14
            case "ResNet50":
                CNNs["ResNet50"]={}
                CNNs["ResNet50"]["model"]=app.config["resnet50"]
                CNNs["ResNet50"]["preprocess_input"]=ResNet50preprocess_input
                CNNs["ResNet50"]["decode_predictions"]=ResNet50decode_predictions
                CNNs["ResNet50"]["last_conv_layer_name"]="conv5_block3_out"
                CNNs["ResNet50"]["scale"]=224 / 7
    #here it iterates for each model
    #then it iterates for each visualization method, so first you process VGG wqith every method, then ResNet with everry method, etc.
    for model in config["models"]:
        print("*************")
        print("model: "+model)
        print("*************")
        runNet(CNNs[model]["model"],image_address,CNNs[model]["preprocess_input"],CNNs[model]["decode_predictions"])
        for method in config["visualizers"].keys(): #checks which method was selected and calls the function with the needed data
            print("*************")
            print("method: "+method)
            print("*************")
            filename=secure_filename(now.strftime("%Y-%m-%d_%H-%M-%S-")+model+"-"+method+"-"+config["filename"])
            match method:
                case 'shap':
                    save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['RESULT_FOLDER'],filename)
                    SHAP(CNNs[model]["model"],image_address,CNNs[model]["preprocess_input"],\
                         CNNs[model]["decode_predictions"], save_path ,\
                         int(config["visualizers"][method]["evals"]),int(config["visualizers"][method]["batch_size"]))
                case 'lime':
                    save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['RESULT_FOLDER'],filename)
                    LIME(CNNs[model]["model"],image_address,CNNs[model]["preprocess_input"],\
                         CNNs[model]["decode_predictions"], save_path ,\
                         int(config["visualizers"][method]["perturbations"]),float(config["visualizers"][method]["kernel_size"]),\
                         float(config["visualizers"][method]["max_dist"]), float(config["visualizers"][method]["ratio"]))
                case 'gradCAM':
                    save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['RESULT_FOLDER'],filename)
                    GRADCAM(CNNs[model]["model"], CNNs[model]["last_conv_layer_name"],image_address,\
                            float(CNNs[model]["scale"]),CNNs[model]["preprocess_input"],\
                            CNNs[model]["decode_predictions"], save_path)
            filenames.append(filename)
    return filenames

