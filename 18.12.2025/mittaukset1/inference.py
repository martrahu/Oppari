import tensorflow as tf
import numpy as np
import joblib

import pathlib

def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
    return file_path


model=tf.keras.models.load_model(get_current_file_path()+'malli.keras')
scaler = joblib.load('scaler.pkl')

def GiveResult(values):

    newSample=[]
    newSample.append([])
    for k in range(len(values[0])):
        for j in range(len(values[0])):
            if k==j:
                continue
            newSample[0].append(values[0][k]/max(values[0][j],1e-5))
    scaled=scaler.transform(newSample)
    
    return model.predict(scaled).argmax()
    #return tf.nn.softmax(model.predict(values)).numpy() #tämä palauttaa jokaisen luokan todennäköisyydet

#test=np.array([[4503,20149,8761,3384]])

#print(GiveResult(test))