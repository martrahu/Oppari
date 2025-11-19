# 19.11.2025

**HumanData.csv** failissa on git:sta [sennoprojekti/SENNOmeasurement/inputData](https://github.com/Sennoharjoittelu/sennoprojekti/tree/main/SENNOmeasurement/inputData) human ja nothuman kansioissa olevat measurement csv failit kasattu yhteen tiedostoon. Y sarakkeessä 1 indikoi human:ia ja 0 nothuman:ia. On poistettu sarakkeet joiden arvot ovat pysyvästi 0 (esim 'S0-value1')

**dnn.py** failissa on malli jossa em dataa käytetään DNN kouluttamiseen. Mallin ylikouluttaminen ei onnistuu hyvin homogeenisen datan johdosta. Alla olevassa kuvassa näkyy  human (Y sarakkeessa 1) ja nothuman esimerkki sampleita. Huomataan että jotkut sarakkeet (esim S1-value2 ja S2-value2) indikoi aika hyvin onko kyseessä
human vai not human.
Mikäli testX täytetään satunnaisarvoilla onnistuu ylikouluttaminen odotetusti.


<img src="../assets/alldata.png">

Alla olevassa korrelaatiotaulukossakin näkyy että S1-value2 ja S2-value2 korreloi kohtalaisen hyvin Y arvon kanssa. Mikäli korrelaatio on sen verran selvä/välitön ei vältämättä tarvitsee neuroverkkoa vaan saisi käyttää myös sklearn classifiereita.
<img src="../assets/corr.png">

Data todennäköisesti tulee olemaan heterogeenisempi kun aletaan ottamaan mittauksia biologisesta kudoksesta.