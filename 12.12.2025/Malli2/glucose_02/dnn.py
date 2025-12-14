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
import numpy as np
from keras.callbacks import  EarlyStopping

noiseAmount=1.25


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

stds=dfAll.std().values
for i in range(len(dfAll.columns)-1):
    rnd = np.random.uniform(low=-stds[i]*noiseAmount, high=stds[i]*noiseAmount,size=(dfAll.shape[0]))
    dfAll[dfAll.columns[i]]+=rnd

dfOneHotencoded = pd.get_dummies(dfAll, columns=['Y'])

X=dfOneHotencoded[dfAll.columns[:len(dfAll.columns)-1]]
Y=pd.DataFrame()
for i in range(len(labelLists)):
    Y['Y_'+str(i)]=dfOneHotencoded['Y_'+str(i)]


## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)
#print(trainY.shape[1])

## Skaalaus
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX.values)
testX=scaler.transform(testX)
joblib.dump(scaler, get_current_file_path()+'Malli2/glucose_02/scaler.pkl')

## Mallin rakennus
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(len(allFiles), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3)
## Mallin kouluttaminen
history=model.fit(
    trainX, trainY,
    validation_data=(testX, testY),  
    epochs=40,
    callbacks=[early_stop],
    batch_size=32,
    verbose=1
)

model.save(get_current_file_path()+'Malli2/glucose_02/malli.keras')

## Mallin evaluaatio
loss, acc=model.evaluate(testX, testY)

print(f"Loss: {loss}")
print(f"Acc: {acc}")


