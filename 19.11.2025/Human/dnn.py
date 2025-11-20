import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import  Adam
from matplotlib import pyplot as plt

## Ladataan tiedosto
df=pd.read_csv('HumanData.csv')

# Erotellaan X ja Y
X = df.drop(['Y'], axis=1) 
Y = df['Y']

## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)

## Skaalaus
scaler = MinMaxScaler()
trainX = scaler.fit_transform(trainX)
testX=scaler.transform(testX)
#testX=np.random.random(testX.shape) #randomilla ylioppiminen tapahtuu odotetusti ja accuracy odotetust 0,5 tasoa

## Mallin rakennus
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

## Mallin kouluttaminen
history=model.fit(
    trainX, trainY,
    #validation_data=(testX, testY),  #kokeilun vuoksi test laitettu validaatioksi
    epochs=50,
    batch_size=8,
    verbose=1
)


"""plt.plot(history.history['loss'],'b',label='trainingLoss')
plt.plot(history.history['val_loss'],'r',label='valLoss')
plt.legend()
plt.show()"""

## Mallin evaluaatio
loss, acc=model.evaluate(testX, testY)

print(f"Loss: {loss}")
print(f"Acc: {acc}")
