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
from sklearn.utils.class_weight import compute_class_weight


def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
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
        dfArr=pd.read_csv('./raw/'+files[i]+'.txt', delimiter= ';')
        dfLabel[dfLabel.columns[i]]=dfArr[dfArr.columns[1]]

    dfLabel['Y']=y

    return dfLabel

labelLists=[]

for i in range(len(allFiles)):
    labelLists.append(AntaaLabelDf(allFiles[i],i))

dfAll=pd.concat(labelLists)

dfAll.dropna(inplace=True)



dfAll.reset_index(drop=True, inplace=True)
#dfAll.drop(['S3','S1','S2'],axis=1, inplace=True)
dfAll.to_csv('AllData.csv', index=False) 

dfOneHotencoded = pd.get_dummies(dfAll, columns=['Y'])

X=dfOneHotencoded[dfAll.columns[:len(dfAll.columns)-1]]
Y=pd.DataFrame()
for i in range(len(labelLists)):
    Y['Y_'+str(i)]=dfOneHotencoded['Y_'+str(i)]


X=dfAll.drop('Y', axis=1)
Y=dfAll['Y']

## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)
#print(trainY.shape[1])

## Skaalaus
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX=scaler.transform(testX)
joblib.dump(scaler, 'scaler.pkl')

## Mallin rakennus
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

#classes = np.unique(trainY)
#classes=np.array([[False,True],[True,False]])
#print(classes)
#class_weights = compute_class_weight('balanced', classes=classes, y=trainY)
#class_weights = dict(zip(classes, class_weights))

## Mallin kouluttaminen
history=model.fit(
    trainX, trainY,
    #class_weight=class_weights,
    validation_data=(testX, testY),  
    epochs=20,
    batch_size=32,
    verbose=1
)

model.save(get_current_file_path()+'mittaus2.keras')

predY=model.predict(testX).round(0).flatten()

print(predY)

## Mallin evaluaatio
"""loss, acc=model.evaluate(testX, testY)

plt.plot(history.history['mae'], 'b', label="train")
plt.plot(history.history['val_mae'], 'r', label="val")
plt.legend()
plt.show()

print(f"Loss: {loss}")
print(f"Acc: {acc}")"""

print("Acc: ",sum(predY==testY)/len(predY))


