
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pathlib
import joblib

def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
    return file_path

allFiles=[]

allFiles.append(['water_gluc50_SLOT_0','water_gluc50_SLOT_1','water_gluc50_SLOT_2','water_gluc50_SLOT_3',])
allFiles.append(['water_gluc100_SLOT_0','water_gluc100_SLOT_1','water_gluc100_SLOT_2','water_gluc100_SLOT_3',])
allFiles.append(['water_pure_SLOT_0','water_pure_SLOT_1','water_pure_SLOT_2','water_pure_SLOT_3',])


dfAll=pd.DataFrame(columns=['S0','S1','S2','S3','Y']) #Lopullinen dataframe jossa mittauksien data yhdessä


def AntaaLabelDf(files, y):
    dfLabel=pd.DataFrame(columns=dfAll.columns)

    for i in range(len(files)):
        dfArr=pd.read_csv(get_current_file_path()+'/raw/'+files[i]+'.txt', delimiter= ';')
        dfLabel[dfLabel.columns[i]]=dfArr[dfArr.columns[1]]

    dfLabel['Y']=y

    return dfLabel

labelLists=[]

for i in range(len(allFiles)):
    labelLists.append(AntaaLabelDf(allFiles[i],i))

dfAll=pd.concat(labelLists)

dfAll = dfAll.dropna()
dfAll.reset_index(drop=True, inplace=True)
#dfAll.drop(['S0','S1','S2'],axis=1, inplace=True)
dfAll.to_csv(get_current_file_path()+'AllData.csv', index=False) 

X=dfAll[dfAll.columns[:len(dfAll.columns)-1]] 
Y=dfAll['Y'] 

## Splitataan data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y)


## Skaalaus
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX=scaler.transform(testX)
joblib.dump(scaler, get_current_file_path()+'scaler.pkl')


## Malli
model = XGBClassifier()
#model = LogisticRegression()

model.fit(trainX, trainY)
preds = model.predict(testX)

## Mallin evaluaatio
print("Accuracy:", accuracy_score(testY, preds))

