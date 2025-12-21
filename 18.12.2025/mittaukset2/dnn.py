import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
import pathlib
from util import GetDfAllSynt
from tensorflow.keras.regularizers import l2

noiseAmount =0.0


def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
    return file_path

dfAll=GetDfAllSynt(noiseAmount)

dfOneHotencoded = pd.get_dummies(dfAll, columns=['Y'])

labelCount=len(dfAll['Y'].unique())

X=dfOneHotencoded[dfAll.columns[:len(dfAll.columns)-1]]
Y=pd.DataFrame()
for i in range(labelCount):
    Y['Y_'+str(i)]=dfOneHotencoded['Y_'+str(i)]

## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)


## Skaalaus
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX.values)
testX=scaler.transform(testX)
joblib.dump(scaler, get_current_file_path()+'scaler.pkl')


## Mallin rakennus
model = Sequential([
    Input(shape=(X.shape[1],)),
    #Dense(8, activation='relu',kernel_regularizer=l2(0.001)),
    #Dense(8, activation='relu',kernel_regularizer=l2(0.001)),
    Dense(8, activation='relu',kernel_regularizer=l2(0.001)),
    Dense(labelCount, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

## Mallin kouluttaminen
history=model.fit(
    trainX, trainY,
    validation_data=(testX, testY),  
    epochs=15,
    batch_size=32,
    verbose=1
)

model.save(get_current_file_path()+'malli.keras')

## Mallin evaluaatio
loss, acc=model.evaluate(testX, testY)

plt.plot(history.history['loss'], 'b', label="train")
plt.plot(history.history['val_loss'], 'r', label="val")
plt.legend()
#plt.show()

print(f"Loss: {loss}")
print(f"Acc: {acc}")
