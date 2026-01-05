**Kansio raw** - mittauksissa taas käytetty syvempää laatikkoa ja mitattauksia otettu vedestä ja vahvemmata sokeriliuoksesta. Alustavasti laatikossa ollut 40ml nestettä, sen jälkeen ruiskulla poistettu 10ml ja sen jälkeen 1 min välein lisätty 1ml nestettä (kts alla oleva kuva). Myöhemmin lisäykset tehty 30sek välein. Datan tallennus nesteen poistaamisen ja lisäyksien ajan ollut tauolla ja vedellä annettu asettautua ennen mittauksien jatkaamista.

<img src="../assets/m2_s0s1.png">


Kuvassa huomataan selvä "portaikko" johtuen veden lisäämisestä ja vastaavasti vedenpinnan noususta. 1ml nesteen lisääminen nostaa ko laatikossa vedenpintaa laskennallisesti noin 0,4mm verran. Ainakin kohtalaisen tarkan etäisyysmittarin Senno laitteesta saisi...


Käytetty samoja DNN hyperparametreja mitä mittaus1 tapauksessa, myös koulutusdata oli ainoastaan synteettiset arvot (kaikki jaettu kaikilla)
Johtuen lähtödatan suuremmasta vaihtelusta tarvitaan DNN kouluttamisessa noin 10 epochia enemmän mitä mittaukset1 kohdalla. Kuitekin 100% tarkkuus nopeasti löytyy.