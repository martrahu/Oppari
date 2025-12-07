
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pathlib
import joblib


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
        dfArr=pd.read_csv('./raw3/'+files[i]+'.txt', delimiter= ';')
        dfLabel[dfLabel.columns[i]]=dfArr[dfArr.columns[1]]

    dfLabel['Y']=y

    return dfLabel

labelLists=[]

for i in range(len(allFiles)):
    labelLists.append(AntaaLabelDf(allFiles[i],i))

dfAll=pd.concat(labelLists)

dfAll = dfAll.dropna()
dfAll.reset_index(drop=True, inplace=True)

dfAll.to_csv('AllData.csv', index=False) 

X=dfAll[dfAll.columns[:len(dfAll.columns)-1]] 
Y=dfAll['Y'] 

## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)


## Skaalaus
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX=scaler.transform(testX)
joblib.dump(scaler, 'scaler.pkl')


## Malli
model = XGBClassifier()
#model = LogisticRegression()

model.fit(trainX, trainY)
preds = model.predict(testX)

## Mallin evaluaatio
print("Accuracy:", accuracy_score(testY, preds))

