import pandas as pd
import pathlib

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

print(labelLists[0].describe())
print(labelLists[1].describe())
print(labelLists[2].describe())
#print((labelLists[0].mean()-labelLists[1].mean())/labelLists[1].mean()*100)
print((labelLists[0].max()-labelLists[0].min())/labelLists[0].mean()*100)