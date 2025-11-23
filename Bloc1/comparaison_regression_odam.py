# # Comparaison des méthodes de régression d'apprentissage supervisé

# Importation des modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Chargement des données
ozone = pd.read_csv("ozonecomplet.csv",header=0, sep=";")

# Nettoyage des données
ozone = ozone.drop(["nomligne", "Dv", "Ne"], axis=1)
ozone.rename(columns={"O3": "Y"}, inplace=True)

# Création d'un dataframe pour comparer les résultats de chaque méthode (n blocs)
PREV = pd.DataFrame(
    {
        "bloc": np.nan,
        "Y": ozone["Y"],
    }
)

# # Entraînement des modèles (avec validation croisée) pour chaque méthode
# Moindres carrés ordinaires, lasso, ridge, elasticNet, arbre, forêt

# Initialisation de KFold
kf = KFold(n_splits=10, shuffle=True, random_state=0)

for i, (train_index, test_index) in enumerate(kf.split(ozone)):

    # Séparation des données en train et test
    X_train = ozone.iloc[train_index].drop(["Y"], axis=1)
    X_test = ozone.iloc[test_index].drop(["Y"], axis=1)
    Y_train = ozone.iloc[train_index]["Y"]

    # Mise à jour de la colonne 'bloc' dans PREV
    PREV.loc[test_index, "bloc"] = i

    # Régression linéaire (MCO)
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    PREV.loc[test_index, "MCO"] = reg.predict(X_test)

    # Lasso avec validation croisée
    kf2 = KFold(n_splits=5, shuffle=True, random_state=0)
    lassocv = LassoCV(cv=kf2)
    pipelassocv = Pipeline(steps=[("cr", StandardScaler()), ("lassocv", lassocv)])
    pipelassocv.fit(X_train, Y_train)
    PREV.loc[test_index, "lasso"] = pipelassocv.predict(X_test)

    # Ridge avec validation croisée
    grilleridge = lassocv.alphas_ * 100
    ridgecv = RidgeCV(cv=kf, alphas=grilleridge)
    piperidgecv = Pipeline(steps=[("cr", StandardScaler()), ("ridgecv", ridgecv)])
    piperidgecv.fit(X_train, Y_train)
    PREV.loc[test_index, "ridge"] = piperidgecv.predict(X_test)

    # ElasticNet avec validation croisée
    grilleelas = lassocv.alphas_ * 2
    elasticcv = ElasticNetCV(cv=kf, alphas=grilleelas)
    pipeelasticcv = Pipeline(steps=[("cr", StandardScaler()), ("elasticcv", elasticcv)])
    pipeelasticcv.fit(X_train, Y_train)
    PREV.loc[test_index, "elas"] = pipeelasticcv.predict(X_test)

    # Arbre de décision
    arbre = DecisionTreeRegressor()
    arbre.fit(X_train, Y_train)
    PREV.loc[test_index, "arbre"] = arbre.predict(X_test)

    # Forêt aléatoire
    foret = RandomForestRegressor(n_estimators=100)
    foret.fit(X_train, Y_train)
    PREV.loc[test_index, "foret100"] = foret.predict(X_test)  
    foret = RandomForestRegressor(n_estimators=500,max_features=0.3)
    foret.fit(X_train, Y_train)
    PREV.loc[test_index, "foret500"] = foret.predict(X_test)

# # Calcul des erreurs pour chaque modèle
Erreur = PREV.copy()
Erreur = Erreur.drop("bloc", axis=1)


def erreur(X, Y):
    return np.mean((X - Y) ** 2)


def apply_erreur(RES):
    return RES.apply(lambda col: erreur(col, RES.iloc[:, 0]), axis=0)


print(apply_erreur(Erreur))