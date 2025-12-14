import pandas as pd
import pathlib
import numpy as np

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

#print(labelLists[0].describe())
#print(labelLists[1].describe())
#print(labelLists[2].describe())
#print((labelLists[0].mean()-labelLists[1].mean())/labelLists[1].mean()*100)
print()
labelLists[2]+=labelLists[2].std()
arr=labelLists[2].std().values

df=labelLists[2]

stds=df.std().values
for i in range(len(df.columns)-1):
    rnd = np.random.uniform(low=-stds[i], high=stds[i],size=(df.shape[0]))
    df[df.columns[i]]+=rnd




"""import tensorflow as tf

penalty_matrix = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],  
        [2.0, 1.0, 0.0] 
    ]

y_true = [[0.0, 0.0, 1.0]]
y_pred = [[0.95, 0.05, 0]]


ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
penalties = tf.matmul(y_true, penalty_matrix)
penalty_weight = tf.reduce_sum(penalties * y_pred, axis=1)

print(ce * (1.0 + penalty_weight))"""


def get_current_file_path():     
    file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())  +'\\'   
    return file_path
print(get_current_file_path())