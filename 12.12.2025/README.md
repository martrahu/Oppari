**Kansio _Mittaukset11.12** - mittaukset 2 eri Senno konfiguraatiolla, Toinen 4 ledia näkyvällä spektrilla ja toinen IR ledit eri asetuksilla. Tähän astiset mallit menee nopeasti testidatalla 100%:iin johtuen labelin sisäisestä vähäisestä vaihtelusta (opetusdata ja testidata on saman mittauksien aikana saatu). Tähän voi olla ratkaisuna mallin karsiminen pienemmänksi tai mittausdatan käsittely ennen neuroverkon kouluttamista, esimerkiksi kohinan lisääminen tai eri reatureiden suhteuden korostaminen jako- tai kertolaskulla. DNN kouluttaminen IR:lla otetulla datasetilla on sisäistä vaihtelevuutta enemmän, siksi DNN kouluttaminen vaatii muutaman epoch kierroksen lisää.

Toiseksi ko ogelmassa on labeleiden järjestyksellä merkitystä - korkean konsentraation määritteleminen puhtaaksi vedeksi on isompi virhe mitä matalan konsentraation määritteleminen vedeksi. Tämä tuo ongelmaan mukaan ominaisuuden joka yleensä on regressiotyylisissä ongelmissa. Tähän voi kokeilla regressio-DNN tai luokittelu DNN jossa kouluttamisessa penalisoidaan "isompaa" virhettä.

Kolmanneksi tähän astisissa mittauksissa näyttää klassifikaattorit pärjäävän stabiilimmin ja nopeammin mitä DNN. Jatkossa DNN toiminnan stabilisoimiseksi voi vakioittaa randomilla generoidut panojen ja biaksen lähtöarvot.

Alla kokeiltavaksi joitakin malleja edellisiin pointteihin liittyen:

**Malli1** - tässä mallissa luokittelu DNN, malli karsittu 2 neuroniin niin että koulutuksessa acc jäi tasolle 80%. Generalisoituuko paremmin?

**Malli2** - luokittelu DNN jossa lähtödataan lisätty random kohinaa parempaa generalisaatiota varten. Muuttuja noiseAmount on maksimaalinen mahdollinen std kertoarvo mikä randomilla mittaukseen lisätään. Tätä arvoa voi kokeilla säätää (muuttamisen jälkeen pitää malli uudestaan kouluttaa).

**Malli3** - regressio DNN.

**Malli4** - sama mikä edellinen mutta noiseAmountilla saa kohinaa lisätä opetusdataan.

**Malli5** - luokittelu DNN jossa kustomoitu loss function joka rankaisee isommasta luokitteluvirheestä enemmän

**Malli6** - sama mikä edellinen mutta noiseAmountilla saa kohinaa lisätä opetusdataan.

**Malli7** - XGB classifier

**Malli8** - sama mikä edellinen mutta noiseAmountilla saa kohinaa lisätä opetusdataan.

**Malli9** - LogReg classifier

**Malli10** - sama mikä edellinen mutta noiseAmountilla saa kohinaa lisätä opetusdataan.

**Malli11** - luokittelu DNN jossa synteettisiä featureita

**Malli12** - sama mikä edellinen mutta noiseAmountilla saa kohinaa lisätä opetusdataan.

Malleissa 2, 4, 6, 8, 10 ja 12 on noiseAmount muuttujaa jonka arvoa voi säätää paremman generalisaation saamiseksi. Malli on muistettava kouluttaa uudestaan mikäli arvoa on muudettu.


**Testaaminen:**

Importataan inferences.py oleva funktio GiveFinalResults():

**from inferences import GiveFinalResults**

GiveFinalResults(conf,truth,sample)

**conf** - arvo joko nr 1 tai 2, riippuen Senno konfiguraatiosta, 1 neljällä ledillä ja 2 kahdella IR ledillä (viittaan Villen discord viestiin 11.12.2025 klo 15:41).

**truth** - joko 0, 1 tai 2 riippuen testattavana olevasta liuoksesta. Tätä arvoa tarvitaan DNN mallien paremmusjärjestykseen laittamiseen.

**sample** - 2D array jossa 4 elementtiä S0, S1, S2 ja S3