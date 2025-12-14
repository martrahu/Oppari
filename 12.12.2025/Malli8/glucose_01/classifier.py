
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pathlib
import joblib
import numpy as np

def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())  +'\\'   
    return file_path

noiseAmount=1
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

dfAll = dfAll.dropna()

dfAll.reset_index(drop=True, inplace=True)
stds=dfAll.std().values
for i in range(len(dfAll.columns)-1):
    rnd = np.random.uniform(low=-stds[i]*noiseAmount, high=stds[i]*noiseAmount,size=(dfAll.shape[0]))
    dfAll[dfAll.columns[i]]+=rnd

X=dfAll[dfAll.columns[:len(dfAll.columns)-1]] 
Y=dfAll['Y'] 

## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)


## Skaalaus
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX=scaler.transform(testX)
joblib.dump(scaler, get_current_file_path()+'Malli8/glucose_01/scaler.pkl')


## Malli
model = XGBClassifier()
#model = LogisticRegression()

model.fit(trainX, trainY)

model.save_model(get_current_file_path()+'Malli8/glucose_01/malli.json')

preds = model.predict(testX)

## Mallin evaluaatio
print("Accuracy:", accuracy_score(testY, preds))

