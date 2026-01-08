***Kansio raw** - Tässä mitattu petrimaljassa alapuolelta kelluvan levyn kanssa. Nestettä  aluksi 20ml, johon lisätty 5ml pisara kerrallaan. Tehty kolme kertaa peräkkäin.

SLOT_3 on molemmassa mittausnäytteessä (vesi ja sokerilios), konstantti arvo 6553200, johtunee sensorin saturoitumisesta kun kovempaa virtaa LED:iin laitettu. S3 jätetty datasta pois.

Koulutuksessa käytetty featuret: S0/S1, S0/S2, S1/S2. Käänteisarvot jätetty pois (S0/S1 on informatiivisesti sama mikä S1/S0).


**dnn1.py** - luokittelu DNN. Yksinkertaisella kokeilulla ei hyvään val_acc päässyt, KerasTuneria käyttämällä saadaan paras tulos 5 syvällä layerilla jossa jokaisessa 16 neuronia (val_acc 0.90 luokkaa)

Histogrameista ei selvästi erottelee vettä ja liosta:

<img source="../assets/hist.png">


Opiskelukäyrä:

<img source="../assets/trainingCurve1.png">



Kokeilin myös alkeellista signaalinkäsittelyä EMA-tyylisellä high-pass filterilla (5Hz rajana). Siitä en mitään järkevää tulosta saanut, mahdollisesti tarvitaan syvempää osaamista DSP:sta...?