# # Comparaison des méthodes de régression d'apprentissage supervisé

# Importation des modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Chargement des données
don =pd.read_csv("SAh.csv",header=0,sep=",")

# Nettoyage des données
don = don.drop(["famhist"], axis=1)
don.rename(columns={"chd": "Y"}, inplace=True)

# Création d'un dataframe pour comparer les résultats de chaque méthode (n blocs)
PROB = pd.DataFrame(
    {
        "bloc": np.nan,
        "Y": don["Y"],
    }
)

# # Entraînement des modèles (avec validation croisée) pour chaque méthode
# Moindres carrés ordinaires, lasso, ridge, elasticNet, arbre, forêt

# Initialisation de KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
#Initialisation des grilles
def grille(X, y, type = "lasso", ng=400):
    scalerX = StandardScaler().fit(X)
    Xcr= scalerX.transform(X)
    l0 = np.abs(Xcr.transpose().dot((y-y.mean()))).max()/X.shape[0]
    llc = np.linspace(0,-4,ng)
    ll = l0*10**llc
    if type=="lasso":
        Cs = 1/ 0.9/ X.shape[0] / (l0*10**(llc))
    elif type=="ridge":
        Cs = 1/ 0.9/ X.shape[0] / ((l0*10**(llc)) * 100)
    elif type=="enet":
        Cs = 1/ 0.9/ X.shape[0] / ((l0*10**(llc)) * 2)
    return Cs

X = don.drop(["Y"], axis=1).to_numpy()
Y = don["Y"].to_numpy()


for app_index, val_index in skf.split(X,Y):
    Xapp = X[app_index,:]
    Xtest = X[val_index,:]
    Yapp = Y[app_index]
    ### regression logistique
    log = LogisticRegression(penalty=None,solver="newton-cholesky").fit(Xapp,Yapp)
    PROB.loc[val_index,"log"] = log.predict_proba(Xtest)[:,1]
    ### arbre
    arbre = DecisionTreeClassifier().fit(Xapp,Yapp)
    PROB.loc[val_index,"arbre"] = arbre.predict_proba(Xtest)[:,1]
    ### foret 100
    foret = RandomForestClassifier(n_estimators=100).fit(Xapp,Yapp)
    PROB.loc[val_index,"foret100"] = foret.predict_proba(Xtest)[:,1]
    ### foret 500
    foret = RandomForestClassifier(n_estimators=500).fit(Xapp,Yapp)
    PROB.loc[val_index,"foret500"] = foret.predict_proba(Xtest)[:,1]
    ### lasso
    cr = StandardScaler()
    Cs_lasso = grille(Xapp,Yapp, "lasso")
    lassocv =  LogisticRegressionCV(cv=10, penalty="l1", n_jobs=10,\
                 Cs=Cs_lasso,  solver="saga", max_iter=2000)
    pipe_lassocv = Pipeline(steps=[("cr", cr), ("lassocv", lassocv)])
    pipe_lassocv.fit(Xapp,Yapp)
    PROB.loc[val_index,"lasso"] = pipe_lassocv.predict_proba(Xtest)[:,1]
    ### elastic net
    cr = StandardScaler()
    Cs_enet = grille(Xapp,Yapp,"enet")
    enetcv=LogisticRegressionCV(cv=10,penalty="elasticnet",n_jobs=10,\
          l1_ratios=[0.5],Cs=Cs_enet,solver="saga",max_iter=2000)
    pipe_enetcv = Pipeline(steps=[("cr", cr), ("enetcv", enetcv)])
    pipe_enetcv.fit(Xapp,Yapp)
    PROB.loc[val_index,"elast"] = pipe_enetcv.predict_proba(Xtest)[:,1] 
    ### ridge
    cr = StandardScaler()
    Cs_ridge = grille(Xapp,Yapp,"ridge")
    ridgecv = LogisticRegressionCV(cv=10, penalty="l2", \
            Cs=Cs_ridge,  max_iter=1000)
    pipe_ridgecv = Pipeline(steps=[("cr", cr), ("ridgecv", ridgecv)])
    pipe_ridgecv.fit(Xapp,Yapp)
    PROB.loc[val_index,"ridge"] = pipe_ridgecv.predict_proba(Xtest)[:,1]

PROB.to_pickle("PROB.pkl")
