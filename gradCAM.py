import numpy as np
import tensorflow as tf
# Display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (6, 6)
#para que elimine las lineas de los ejes deberia de cambiar cada uno a False por separado
plt.axis('off')
import cv2

image_path = "./tesina-imagenes/"
img_base_path = image_path + "1002.jpg"

def GRADCAM(model, last_conv_layer,image,scale,preprocessor, decoder, savefile_path):
  from scipy.ndimage import zoom

  def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
      # First, we create a model that maps the input image to the activations
      # of the last conv layer as well as the output predictions
      grad_model = tf.keras.models.Model(
          [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
      )

      # Then, we compute the gradient of the top predicted class for our input image
      # with respect to the activations of the last conv layer
      with tf.GradientTape() as tape:
          last_conv_layer_output, preds = grad_model(img_array)
          if pred_index is None:
              pred_index = tf.argmax(preds[0])
          class_channel = preds[:, pred_index]

      # This is the gradient of the output neuron (top predicted or chosen)
      # with regard to the output feature map of the last conv layer
      grads = tape.gradient(class_channel, last_conv_layer_output)

      # This is a vector where each entry is the mean intensity of the gradient
      # over a specific feature map channel
      pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
      #print(pooled_grads)

      # We multiply each channel in the feature map array
      # by "how important this channel is" with regard to the top predicted class
      # then sum all the channels to obtain the heatmap class activation

      last_conv_layer_output = last_conv_layer_output[0]
      heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
      heatmap = tf.squeeze(heatmap)

      # For visualization purpose, we will also normalize the heatmap between 0 & 1
      # ademas, al utilizar el tf.maximum(heatmap,0) seria la parte del ReLU
      # Ya que el Relu se tiene que aplicar sobre el resultado de la combinacion lineal de los mapas de activaciones
      # siendo ese resultado el heatmap calculado en las lineas previas
      # nos interesa la influencia positiva en la clase de interes
      # pero esto es solo para la representacion visual final
      heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

      global conv
      global weights
      weights = pooled_grads[..., tf.newaxis]
      conv=np.array(last_conv_layer_output)
      return heatmap.numpy()

  conv=[]
  weights=[]

  img = cv2.imread(image)
  img = cv2.resize(img,(224,224))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  X = np.expand_dims(img, axis=0).astype(np.float32)
  X = preprocess_input(X)

  np.random.seed(222)

  # Print what the top predicted class is
  preds = model.predict(X,verbose=0)
  print("Predicted:", decode_predictions(preds, top=1)[0])
  result = decode_predictions(preds)
  for res in result[0]:
    print(res)

  # Remove last layer's softmax
  model.layers[-1].activation = None
  # Generate class activation heatmap
  heatmap = make_gradcam_heatmap(X, model, last_conv_layer)

  # Display heatmap
  #plt.matshow(heatmap)
  #plt.show()

  """
  def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.5):
      # Load the original image
      img = keras.utils.load_img(img_path)
      img = keras.utils.img_to_array(img)

      # Rescale heatmap to a range 0-255
      heatmap = np.uint8(255 * heatmap)

      # Use jet colormap to colorize heatmap
      jet = cm.get_cmap("jet")

      # Use RGB values of the colormap
      jet_colors = jet(np.arange(256))[:, :3]
      jet_heatmap = jet_colors[heatmap]

      # Create an image with RGB colorized heatmap
      jet_heatmap = keras.utils.array_to_img(jet_heatmap)
      jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
      jet_heatmap = keras.utils.img_to_array(jet_heatmap)

      # Superimpose the heatmap on original image
      superimposed_img = jet_heatmap * alpha + img
      superimposed_img = keras.utils.array_to_img(superimposed_img)

      # Save the superimposed image
      superimposed_img.save(cam_path)

      # Display Grad CAM
      display(Image(cam_path))
  """

  plt.figure(figsize=(12, 6))
  a = plt.imshow(img)
  plt.imshow(zoom(heatmap, zoom=(scale, scale)), cmap='jet', alpha=0.5)
  plt.savefig(savefile_path,bbox_inches='tight')
  print("image saved at: "+savefile_path)
  plt.clf()


seleccion=2
if seleccion == 1:
  from keras.applications.vgg16 import preprocess_input,decode_predictions
  last_conv_layer_name = "block5_conv3"
  scale = 224 / 14
elif seleccion == 2:
  from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
  last_conv_layer_name = "conv5_block3_out"
  scale = 224 / 7
else:
  print("elegir 1 o 2")

"""image_path = "./static/files/results"
for name in namelistImagenet:
  img_base_path = image_path + name
  ubicacion="imagenet/GRADCAMRes"+name
  #GRADCAM(tf.keras.applications.VGG16(weights="imagenet"),"block5_conv3",img_base_path,224/14,preprocess_input,decode_predictions, ubicacion)
  GRADCAM(tf.keras.applications.ResNet50(weights="imagenet"),last_conv_layer_name,img_base_path,scale,preprocess_input,decode_predictions, ubicacion)
  """
