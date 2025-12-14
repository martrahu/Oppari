import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.saving import register_keras_serializable
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from loss import CostSensitiveLoss


def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())  +'\\'   
    return file_path


allFiles=[]

# Jokaisen luokan raw failit ovat omassa listassa
allFiles.append(['water_gluc50_SLOT_0','water_gluc50_SLOT_1','water_gluc50_SLOT_2','water_gluc50_SLOT_3',])
allFiles.append(['water_gluc100_SLOT_0','water_gluc100_SLOT_1','water_gluc100_SLOT_2','water_gluc100_SLOT_3',])
allFiles.append(['water_pure_SLOT_0','water_pure_SLOT_1','water_pure_SLOT_2','water_pure_SLOT_3',])


dfAll=pd.DataFrame(columns=['S0','S1','S2','S3','Y']) #Lopullinen dataframe jossa mittauksien data yhdessä


def AntaaLabelDf(files, y):
    dfLabel=pd.DataFrame(columns=dfAll.columns)

    for i in range(len(files)):
        dfArr=pd.read_csv(get_current_file_path()+'/raw/glucose_01/'+files[i]+'.txt', delimiter= ';')
        dfLabel[dfLabel.columns[i]]=dfArr[dfArr.columns[1]]

    dfLabel['Y']=y

    return dfLabel

labelLists=[]

for i in range(len(allFiles)):
    labelLists.append(AntaaLabelDf(allFiles[i],i))

dfAll=pd.concat(labelLists)

dfAll.dropna(inplace=True)

dfAll.reset_index(drop=True, inplace=True)

dfOneHotencoded = pd.get_dummies(dfAll, columns=['Y'])

X=dfOneHotencoded[dfAll.columns[:len(dfAll.columns)-1]]
Y=pd.DataFrame()
for i in range(len(labelLists)):
    Y['Y_'+str(i)]=dfOneHotencoded['Y_'+str(i)]


## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)

## Skaalaus
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX.values)
testX=scaler.transform(testX)
joblib.dump(scaler, get_current_file_path()+'Malli5/glucose_01/scaler.pkl')

## Mallin rakennus
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(len(allFiles), activation='softmax')
])

penalty_matrix = tf.constant(
    [
        [0.0, 1.0, 3.0], 
        [1.0, 0.0, 1.0], 
        [3.0, 1.0, 0.0],  
    ],
    dtype=tf.float32
)


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CostSensitiveLoss(penalty_matrix),
    metrics=['accuracy']
)

## Mallin kouluttaminen
history=model.fit(
    trainX, trainY,
    validation_data=(testX, testY),  
    epochs=7,
    batch_size=32,
    verbose=1
)

model.save(get_current_file_path()+'Malli5/glucose_01/malli.keras')

## Mallin evaluaatio
loss, acc=model.evaluate(testX, testY)

print(f"Loss: {loss}")
print(f"Acc: {acc}")


