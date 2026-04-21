# Projet Data_python

*Par  EDDARDOURI Houda, FERRAND-BASTET Adrien, ID-BELLA Sara

# Table des matières
1. [Définitions](#definitions)
2. [Objectifs](#objectifs)
3. [Sources des données](#sources)
4. [Présentation du dépôt](#pres)
5. [Méthodologie](#meth)

## 1. Définitions <a name="definitions">

**gold** : prix de l’or
Prix du métal précieux sur les marchés financiers (USD/once). Actif refuge utilisé en période d’incertitude. Variable cible du modèle.

**dxy** : US Dollar Index
Indice mesurant la force du dollar face à un panier de devises. Généralement inversement corrélé au prix de l’or.

**sp500** : indice S&P 500
Indice des 500 plus grandes entreprises américaines. Reflète le marché actions et le sentiment global de risque.

**vix** : indice CBOE Volatility Index (VIX)
Mesure la volatilité anticipée du marché. Indicateur de stress financier (“indice de la peur”).

**cpi** : inflation (Consumer Price Index)
Indice mesurant l’évolution des prix à la consommation. Indicateur clé de l’inflation et des politiques monétaires.

**gpr** : indice de risque géopolitique
Indice basé sur l’analyse de la presse mesurant les tensions géopolitiques. Capture les chocs exogènes affectant les marchés.

## 2. Objectifs <a name="objectifs">

L’objectif de ce projet est de modéliser et prédire le prix de l’or à partir de variables financières et macroéconomiques définies précédement.

## 3. Sources des données <a name="sources">

Les données utilisées dans ce projet proviennent de sources publiques et sont récupérées de différentes manières :

    - Yahoo Finance : données financières récupérées via l’API Python yfinance (prix de l’or, DXY, S&P 500, VIX)
    - Federal Reserve : Bank of St. Louis: données macroéconomiques récupérées via l’API REST FRED (inflation – CPI)
    - Matteo Iacoviello : indice GPR récupéré par web scraping (téléchargement automatique du fichier Excel depuis le site officiel)

## 4. Présentation du dépôt <a name=pres>

Notre production est essentiellement localisée dans deux versions d'un fichier ```main.ipynb```.
- La première ne contient que le code non exécuté et les commentaires entre les cellules. 
- Le code dans la seconde a été préalablement exécuté, afin de pouvoir présenter également les résultats. Certaines visualisations ne s’affichent pas  dans la prévisualisation GitHub du notebook, car elles sont générées avec en mode interactif, un format qui n’est pas entièrement pris en charge par GitHub.

C'est cette version exécutée qui tient lieu de rapport final.

Le dossier ```data``` contient une copie locale (csv et xls) des données que nous utilisons.

Le dossier ```scripts``` contient, comme on l'imagine, une multitude de fonctions utiles, afin de rendre notre code plus lisible et maintenanble. 

Quant au fichier ```requirements.txt```, il est appelé par pip afin d'installer les paquets nécessaires en début d'exécution.

## 5. Méthodologie <a name="meth">
Le projet suit les étapes principales suivantes :

    - Collecte des données
    - Préprocessing des séries
    - Analyse exploratoire
    - Tests statistiques
    - Modélisation VAR
    - Extension en VAR-GARCH-X
