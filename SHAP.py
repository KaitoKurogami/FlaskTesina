import json
import numpy as np
import shap
import copy
# Display
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
#para que elimine las lineas de los ejes deberia de cambiar cada uno a False por separado
plt.axis('off')
import cv2

def SHAP(model,image,preprocessor,decoder, savefile_path ,SHAP_max_evals=1000,SHAP_batch_size=50):

#preprocess image
  img = image
  img = cv2.imread(img)
  img = cv2.resize(img,(224,224))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



  url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
  with open(shap.datasets.cache(url)) as file:
      class_names = [v[1] for v in json.load(file).values()]

  X = np.expand_dims(img, axis=0).astype(np.float32)
  Xi=copy.deepcopy(X).astype(int)
  preprocess_input=preprocessor
  X = preprocess_input(X)

  preds = model.predict(X,verbose=0)
  #print("Predicted:", decode_predictions(preds, top=1)[0])
  result = decoder(preds)
  for res in result[0]:
    print(res)

  # python function to get model output; replace this function with your own model function.
  def f(x):
      return model.predict(x,verbose=0)

  # define a masker that is used to mask out partitions of the input image.
  masker_blur = shap.maskers.Image("blur(224,224)", X[0].shape)

  # create an explainer with model and image masker
  explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)

  # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
  shap_values = explainer_blur(X, max_evals=SHAP_max_evals, batch_size=SHAP_batch_size, outputs=shap.Explanation.argsort.flip[:1])

  #return shap_values
  shap_values.data=Xi
  plot=shap.image_plot(shap_values[0],show=False)
  #plot=shap.image_plot(shap_values)
  plt.savefig(savefile_path,bbox_inches='tight')
  print("image saved at: "+savefile_path)
  plt.clf()
  #return(f)

"""
image_path = "./tesina-imagenes/"
#namelist=["EntleBucher.jpg","dutch.jpg","Macaw.jpg","Maze.jpg","forklift.jpg","toucan.jpg","weasel.jpg"]
#namelist=["abacus.jpg","baseball.jpg","dumbbell.jpg","microphone.jpg","tostadora2.jpg"]
namelist=["1002.jpg"]
for name in namelist:
  img_base_path = image_path + name
  ubicacion="resultados/random/"+"SHAPRes"+name
  SHAP(tf.keras.applications.ResNet50(weights="imagenet"),img_base_path,preprocess_input,decode_predictions, ubicacion ,SHAP_max_evals=1000)
"""
