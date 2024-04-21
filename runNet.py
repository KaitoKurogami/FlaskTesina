import cv2
import numpy as np

def runNet(model,image,preprocessor, decoder):
    decode_predictions=decoder

    img = cv2.imread(image)
    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    X = np.expand_dims(img, axis=0).astype(np.float32)
    preprocess_input=preprocessor
    X = preprocess_input(X)

    preds = model.predict(X,verbose=0)
    result = decode_predictions(preds)

    for res in result[0]:
        print(f"clase detectada: {res[1]}, con probabilidad: {res[2]:.3%}")
        #print(txt.format(classification=res[1],probability=res[2]*100.0))