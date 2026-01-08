import tensorflow as tf
import numpy as np
import joblib

import pathlib

def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
    return file_path


model=tf.keras.models.load_model(get_current_file_path()+'malli1.keras')
scaler = joblib.load(get_current_file_path()+'scaler.pkl')

def GiveResult(values):

    temp=values[0][:3]
    values=[temp]

    newSample=[]
    newSample.append([])
    for k in range(len(values[0])):
        for j in range(k,len(values[0])):
            if k==j:
                continue
            newSample[0].append(values[0][k]/max(values[0][j],1e-5))
    
    

    scaled=scaler.transform(newSample)
    
    return model.predict(scaled).argmax()

test=np.array([[4503,20149,8761,3384]])

print(GiveResult(test))