**kansio raw2** - mitattu 8 eri väriä, laitetta liikutettu

**raw3** - 5 väriä raw kansiosta ja puuttuvat raw2 kansiosta (mittaukset otettu eri päivinä). 

(Tulokset ovat samanlaisia riippumata päivästä jolloin mittauksia on otettu.)

**dnn2.py** - neuroverkon kouluttamisfaili. Accuracy tulokset ovat luokkaa ~92%, vaihtelevuus +/- 3%. 

**inference.py** - saa testata koulutettua mallia (joka tallennetaan dnn failissa opetuksen jälkeen). Esim AllData.csv failista saa ottaa kokeilua varten näytteitä.

Alla olevassa kuvassa opiskelukäyrä. Taas huomataan että käytettävä data on hyvin homogeeninen, mallia ei saa ylioppimaan (classifierieilla voisi saada parempia tuloksia)
<img src="./assets/learningCurve.png">



**classifier.py** - tässä failissa kokeilin XGBClassifier:ia sekä LogisticRegression classifieria. Molemmalla saadaan tarkkuus tasolle 98%, vaihtelevuutta vähemmän mitä DNN:ssa. LogisticRegressio mallin hyvä suoriutuminen on merkki lähtödatan ja lopputuloksen lineaarisesta suhteesta (=ei tarvitsee eikä ole optimaalista käyttää neuroverkkoa).
