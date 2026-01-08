import pandas as pd
import pathlib
import numpy as np


def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
    return file_path

def GetDfAll(noiseAmount=0):

    allFiles=[]

    #Jokaisen labelin failit omalla appendilla
    allFiles.append(['below_water_SLOT_0','below_water_SLOT_1','below_water_SLOT_2','below_water_SLOT_3'])
    allFiles.append(['below_glucose_SLOT_0','below_glucose_SLOT_1','below_glucose_SLOT_2','below_glucose_SLOT_3'])

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
            
    dfAll.drop('S3',axis=1,inplace=True)

    return dfAll


def GetDfAllSynt(noiseAmount=0,df=None ):
    if not isinstance(df, pd.DataFrame):
        dfAll=GetDfAll(noiseAmount)
    else:
        dfAll=df

    dfNew=pd.DataFrame()
    for i in range(len(dfAll.columns)-1):
        for j in range(i,len(dfAll.columns)-1):
            if i==j:
                continue
            dfNew['S'+str(i)+'/S'+str(j)]=dfAll['S'+str(i)]/dfAll['S'+str(j)]

    dfNew['Y']=dfAll['Y']
    
    #dfNew.to_csv("AllDataSynt.csv", index=False)
    return dfNew

def GetHPFiltered(noiseAmount=0):

    dfAll=GetDfAll(noiseAmount)

    lbls=[]
    lbls.append(dfAll[dfAll['Y']==0])
    lbls.append(dfAll[dfAll['Y']==1])

    """
    High-pass filter via exponential moving average subtraction

    signal : 1D numpy array
    fs     : sampling rate (Hz)
    fc     : cutoff frequency (Hz)
    """
    fs=2458 
    fc=5
    dt = 1.0 / fs
    tau = 1.0 / (2 * np.pi * fc)
    beta = dt / (tau + dt)
    #print(dfAll.columns[0])

    for j in range(2):
        for k in range(3): 
            signal=lbls[j][lbls[j].columns[k]]
            avg = 0.0
            y = np.zeros_like(signal, dtype=float)

            for i, x in enumerate(signal):
                avg += beta * (x - avg)
                y[i] = x - avg


            lbls[j][lbls[j].columns[k]]=y

    dfNew=pd.concat(lbls)
    
    dfNew=GetDfAllSynt(noiseAmount,dfNew)

    return dfNew
