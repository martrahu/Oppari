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

def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
    return file_path


allFiles=[]

# Jokaisen luokan raw failit ovat omassa listassa
allFiles.append(['black_SLOT_0','black_SLOT_1','black_SLOT_2','black_SLOT_3',]) #Mustan labelin failit (mitattu mustaa vasten). Mustan label on 0
allFiles.append(['blue_SLOT_0','blue_SLOT_1','blue_SLOT_2','blue_SLOT_3',]) #Sinisen labelin failit (mitattu sinistä vasten). Sinisen label on 1
allFiles.append(['green_SLOT_0','green_SLOT_1','green_SLOT_2','green_SLOT_3',]) #Label on 2
allFiles.append(['white_SLOT_0','white_SLOT_1','white_SLOT_2','white_SLOT_3',]) #Label on 3
allFiles.append(['yellow_SLOT_0','yellow_SLOT_1','yellow_SLOT_2','yellow_SLOT_3',]) #jne
allFiles.append(['red_SLOT_0','red_SLOT_1','red_SLOT_2','red_SLOT_3',])
allFiles.append(['superred_SLOT_0','superred_SLOT_1','superred_SLOT_2','superred_SLOT_3',])
allFiles.append(['brightgreen_SLOT_0','brightgreen_SLOT_1','brightgreen_SLOT_2','brightgreen_SLOT_3',])


dfAll=pd.DataFrame(columns=['S0','S1','S2','S3','Y']) #Lopullinen dataframe jossa mittauksien data yhdessä


def AntaaLabelDf(files, y):
    dfLabel=pd.DataFrame(columns=dfAll.columns)

    for i in range(len(files)):
        dfArr=pd.read_csv(get_current_file_path()+'/raw2/'+files[i]+'.txt', delimiter= ';')
        dfLabel[dfLabel.columns[i]]=dfArr[dfArr.columns[1]]

    dfLabel['Y']=y

    return dfLabel

labelLists=[]

for i in range(len(allFiles)):
    labelLists.append(AntaaLabelDf(allFiles[i],i))

dfAll=pd.concat(labelLists)

dfAll.dropna(inplace=True)

dfAll.reset_index(drop=True, inplace=True)

dfAll.to_csv('AllData.csv', index=False) 

dfOneHotencoded = pd.get_dummies(dfAll, columns=['Y'])

X=dfOneHotencoded[dfAll.columns[:len(dfAll.columns)-1]]
Y=pd.DataFrame()
for i in range(len(labelLists)):
    Y['Y_'+str(i)]=dfOneHotencoded['Y_'+str(i)]

## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)


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
    Dense(len(allFiles), activation='softmax')
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
    epochs=20,
    batch_size=32,
    verbose=1
)

model.save(get_current_file_path()+'mittaus2.keras')

## Mallin evaluaatio
loss, acc=model.evaluate(testX, testY)

plt.plot(history.history['loss'], 'b', label="train")
plt.plot(history.history['val_loss'], 'r', label="val")
plt.legend()
plt.show()

print(f"Loss: {loss}")
print(f"Acc: {acc}")


