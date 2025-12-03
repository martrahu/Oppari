import tensorflow as tf
import numpy as np
import joblib

# Black - 0, Blue - 1, Green - 2, White - 3, Yellow - 4

model=tf.keras.models.load_model('mittaus1.keras')
scaler = joblib.load('scaler.pkl')

def GiveResult(values):
    values=scaler.transform(values)
    
    return model.predict(values).argmax()
    #return tf.nn.softmax(model.predict(values)).numpy() #tämä palauttaa jokaisen luokan todennäköisyydet

#test=np.array([[4503,20149,8761,3384]])

#print(GiveResult(test))