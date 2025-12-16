import tensorflow as tf
import numpy as np
import joblib
import pathlib
from loss import CostSensitiveLoss
import xgboost as xgb


def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.resolve())  +'\\'   
    return file_path

def GiveFinalResults(conf,truth,sample):

    resultDictDNN={}
    resultDictclas={}

    for i in range(1,13):

        scaler = joblib.load(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/scaler.pkl')
        scaled=scaler.transform(sample)
        if i>=3 and i<=4:
            model=tf.keras.models.load_model(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/malli.keras')
            res=model.predict(scaled)
            res=np.array([[res[0][0],0,0]])
            resultDictDNN['Malli'+str(i)]=[res.argmax(),res[0][truth]/sum(res[0])]
        elif i<=6:
            model=tf.keras.models.load_model(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/malli.keras')
            res=model.predict(scaled)
            resultDictDNN['Malli'+str(i)]=[res.argmax(),res[0][truth]/sum(res[0])]
        elif i>=7 and i<=8:
            model = xgb.XGBClassifier()
            model.load_model(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/malli.json')
            res=model.predict(scaled)
            resultDictclas['Malli'+str(i)]=res[0]
        elif i>=9 and i<=10:
            model=joblib.load(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/malli.lg')
            res=model.predict(scaled)
            resultDictclas['Malli'+str(i)]=res[0]
        else:
            model=tf.keras.models.load_model(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/malli.keras')

            newSample=sample.copy()

            newSample[0][0]=sample[0][1]/max(sample[0][0],1e-5)
            newSample[0][1]=sample[0][3]/max(sample[0][2],1e-5)
            newSample[0][2]=sample[0][1]/max(sample[0][3],1e-5)
            newSample[0][3]=sample[0][2]/max(sample[0][0],1e-5)

            scaled=scaler.transform(newSample)
            res=model.predict(scaled)
            resultDictDNN['Malli'+str(i)]=[res.argmax(),res[0][truth]/sum(res[0])]

    resultDictDNN = {k: v for k, v in sorted(resultDictDNN.items(), key=lambda item: item[1][1], reverse=True)}

    print('\nLuokitelu DNN Tulosjärjestys:')
    regStr=""
    for e in resultDictDNN:
        if e=='Malli3' or e=='Malli4':
            regStr+=e+ ' tulos: '+str(resultDictDNN[e][0])+ '  truth prob:'+str(resultDictDNN[e][1])+'\n'
        else:
            print(e, 'tulos:',resultDictDNN[e][0], '  truth prob:',resultDictDNN[e][1])

    print('\nRegressio tulokset:')
    print(regStr)
    print('\nClassifier tulokset:')




    for e in resultDictclas:
        print(e, 'tulos:',resultDictclas[e])
    print()




test=np.array([[241512,125158,327660,32545]])

GiveFinalResults(1,2,test)

