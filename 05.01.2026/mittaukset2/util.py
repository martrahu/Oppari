import pandas as pd
import pathlib
import numpy as np

def GetDfAll(noiseAmount=0):

    def get_current_file_path():     
        file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
        return file_path

    allFiles=[]

    #Jokaisen labelin failit omalla appendilla
    allFiles.append(['bb_water_SLOT_0','bb_water_SLOT_1','bb_water_SLOT_2','bb_water_SLOT_3'])
    allFiles.append(['bb_glucose_1m_SLOT_0','bb_glucose_1m_SLOT_1','bb_glucose_1m_SLOT_2','bb_glucose_1m_SLOT_3'])

    #dfAll=pd.DataFrame(columns=['S0','S1','S2','S3','Y']) #Lopullinen dataframe jossa mittauksien data yhdessä
    dfAll=pd.DataFrame() 
    for i in range(len(allFiles[0])):
        dfAll['S'+str(i)] = None

    def AntaaLabelDf(files, y):
        dfLabel=pd.DataFrame(columns=dfAll.columns)

        for i in range(len(files)):
            dfArr=pd.read_csv(get_current_file_path()+'/raw/'+files[i]+'.txt', delimiter= ';',header=None)
            dfLabel[dfLabel.columns[i]]=dfArr[dfArr.columns[1]]

        dfLabel['Y']=y
        return dfLabel

    labelLists=[]

    for i in range(len(allFiles)):
        labelLists.append(AntaaLabelDf(allFiles[i],i))

    dfAll=pd.concat(labelLists)

    #dfAll.to_csv("AllData.csv", index=False)

    dfAll.dropna(inplace=True)

    dfAll.reset_index(drop=True, inplace=True)

    if noiseAmount>0:
        stds=dfAll.std().values
        for i in range(len(dfAll.columns)-1):
            rnd = np.random.uniform(low=-stds[i]*noiseAmount, high=stds[i]*noiseAmount,size=(dfAll.shape[0]))
            dfAll[dfAll.columns[i]]+=rnd

    return dfAll

def GetDfAllSynt(noiseAmount=0):
    dfAll=GetDfAll(noiseAmount)

    dfNew=pd.DataFrame()
    for i in range(len(dfAll.columns)-1):
        for j in range(len(dfAll.columns)-1):
            if i==j:
                continue
            dfNew['S'+str(i)+'/S'+str(j)]=dfAll['S'+str(i)]/dfAll['S'+str(j)]

    dfNew['Y']=dfAll['Y']
    
    #dfNew.to_csv("AllDataSynt.csv", index=False)

    return dfNew
