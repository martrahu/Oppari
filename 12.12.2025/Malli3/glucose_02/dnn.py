import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.metrics import  AUC
import joblib
import matplotlib.pyplot as plt
import pathlib
import numpy as np



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
        dfArr=pd.read_csv(get_current_file_path()+'/raw/glucose_02/'+files[i]+'.txt', delimiter= ';')
        dfLabel[dfLabel.columns[i]]=dfArr[dfArr.columns[1]]

    dfLabel['Y']=y

    return dfLabel

labelLists=[]

for i in range(len(allFiles)):
    labelLists.append(AntaaLabelDf(allFiles[i],i))

dfAll=pd.concat(labelLists)

dfAll.dropna(inplace=True)

dfAll.reset_index(drop=True, inplace=True)

X=dfAll.drop('Y', axis=1)
Y=dfAll['Y']

## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)

## Skaalaus
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX.values)
testX=scaler.transform(testX)
joblib.dump(scaler, get_current_file_path()+'Malli3/glucose_02/scaler.pkl')

## Mallin rakennus
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae',AUC(name='auc')]
)


## Mallin kouluttaminen
history=model.fit(
    trainX, trainY,
    validation_data=(testX, testY),  
    epochs=20,
    batch_size=32,
    verbose=1
)

model.save(get_current_file_path()+'Malli3/glucose_02/malli.keras')

predY=np.clip(model.predict(testX).round(0),0,len(labelLists)-1).flatten()

## Mallin evaluaatio
print("Acc: ",sum(predY==testY)/len(predY))


