<<<<<<< HEAD
import numpy as np
import copy
=======
import tensorflow as tf
import numpy as np
import copy
from tensorflow import keras
#from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#from keras.applications.vgg16 import preprocess_input,decode_predictions
>>>>>>> 6b2678115cb79fae35995b9fa33fda84ad5abc96
import skimage.io
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from PIL import Image

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (6, 6)
#para que elimine las lineas de los ejes deberia de cambiar cada uno a False por separado
plt.axis('off')

import cv2

#image_path = "./tesina-imagenes/"
#img_base_path = image_path + "perro.jpg"

#print('Notebook running: keras ', keras.__version__)
np.random.seed(222)


def LIME(model,image,preprocessor, decoder, savefile_path, LIME_num_perturb,LIME_kernel_size=2.5,LIME_max_dist=224/8, LIME_ratio=0.3):

  num_perturb=LIME_num_perturb

  decode_predictions=decoder

#preprocess image, data from cv2 is to be processed by the CNN
  img = image
  img = cv2.imread(img)
  img = cv2.resize(img,(224,224))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  Xi=copy.deepcopy(img)

#images from skimage are to make te superpixels and to show results graphically
  X2 = skimage.io.imread(image)
  X2 = skimage.transform.resize(X2, (224,224))

  X = np.expand_dims(img, axis=0).astype(np.float32)
  preprocess_input=preprocessor
  X = preprocess_input(X)

  preds = model.predict(X,verbose=0)
  #print("Predicted:", decode_predictions(preds, top=1)[0])
  result = decode_predictions(preds)
  for res in result[0]:
    print(res)

  top_pred_classes = preds[0].argsort()[-5:][::-1]
  top_pred_classes                #Index of top 5 classes

  superpixels = skimage.segmentation.quickshift(X2, kernel_size=LIME_kernel_size,max_dist=LIME_max_dist, ratio=LIME_ratio)
  num_superpixels = np.unique(superpixels).shape[0]
  print(str(num_superpixels)+" superpixeles")

  skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi, superpixels))

  perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

  def perturb_image(img,perturbation,segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    return perturbed_image


  predictions = []
  for pert in tqdm(perturbations):
    perturbed_img = perturb_image(Xi,pert,superpixels)
    perturbed_img = np.expand_dims(perturbed_img, axis=0).astype(np.float32)
    perturbed_img = preprocess_input(perturbed_img)
    pred = model.predict(perturbed_img,verbose=0)
    result = decode_predictions(pred)
    #for res in result[0]:
      #print(res)
    predictions.append(pred)

  predictions = np.array(predictions)

  original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled
  distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
  distances.shape

  kernel_width = 0.25
  weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
  weights.shape

  class_to_explain = top_pred_classes[0]
  simpler_model = LinearRegression()
  simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
  coeff = simpler_model.coef_[0]

  if int(num_superpixels*0.1)<7:
    num_top_features=7
  else:
    num_top_features=int(num_superpixels*0.1)
  top_features = np.argsort(coeff)[-num_top_features:]
  top_features

  mask = np.zeros(num_superpixels)
  mask[top_features]= True #Activate top superpixels


  #return perturb_image(X2,mask,superpixels)

  imagen_resultado=perturb_image(X2,mask,superpixels)
  skimage.io.imsave(savefile_path,np.array(Image.fromarray((imagen_resultado * 255).astype(np.uint8))))

  print("image saved at: "+savefile_path)


<<<<<<< HEAD
"""
=======
>>>>>>> 6b2678115cb79fae35995b9fa33fda84ad5abc96
image_path = "./tesina-imagenes/random/"
#namelist=["EntleBucher.jpg","dutch.jpg","Macaw.jpg","Maze.jpg","forklift.jpg","toucan.jpg","weasel.jpg"]
namelist=["abacus.jpg","baseball.jpg","dumbbell.jpg","microphone.jpg","tostadora2.jpg"]
#namelist=["abacus.jpg"]
for name in namelist:
  img_base_path = image_path + name
  ubicacion="Resultados/random/"+"LIMERes"+name
<<<<<<< HEAD
  LIME(tf.keras.applications.ResNet50(weights="imagenet"),img_base_path,preprocess_input,decode_predictions, ubicacion,1000,LIME_max_dist=224/8)
"""
=======
  LIME(tf.keras.applications.ResNet50(weights="imagenet"),img_base_path,preprocess_input,decode_predictions, ubicacion,1000,LIME_max_dist=224/8)
>>>>>>> 6b2678115cb79fae35995b9fa33fda84ad5abc96
