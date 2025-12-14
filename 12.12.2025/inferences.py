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

    for i in range(1,11):

        scaler = joblib.load(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/scaler.pkl')
        scaled=scaler.transform(sample)
        if i<=6:
            model=tf.keras.models.load_model(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/malli.keras')
            res=model.predict(scaled)
            resultDictDNN['Malli'+str(i)]=[res.argmax(),res[0][truth]/sum(res[0])]
        elif i>=7 and i<=8:
            model = xgb.XGBClassifier()
            model.load_model(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/malli.json')
            res=model.predict(scaled)
            resultDictclas['Malli'+str(i)]=res[0]
        else:
            model=joblib.load(get_current_file_path()+'Malli'+str(i)+'/glucose_0'+str(conf)+'/malli.lg')
            res=model.predict(scaled)
            resultDictclas['Malli'+str(i)]=res[0]

    resultDictDNN = {k: v for k, v in sorted(resultDictDNN.items(), key=lambda item: item[1][1], reverse=True)}

    print('\nDNN Tulosjärjestys:')
    for e in resultDictDNN:
        print(e, 'tulos:',resultDictDNN[e][0], '  truth prob:',resultDictDNN[e][1])

    print('\nClassifier tulokset:')

    for e in resultDictclas:
        print(e, 'tulos:',resultDictclas[e])
    print()




#test=np.array([[241512,125158,327660,32545]])

#GiveFinalResults(1,0,test)

