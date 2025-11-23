import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
mat = pd.read_csv('../DONNEES/mat.txt', header=None, index_col=0, sep=';')

# Définir la grille
grille = list(range(2, 10)) + list(range(10, 91, 10)) + list(range(100, 1001, 100))

# Initialiser le résultat
res = []

# Calculer le rapport max/min des normes
for g in grille:
    tmp = np.linalg.norm(mat.iloc[:, :g], axis=1)
    res.append(max(tmp) / min(tmp))

# Tracer l'évolution du contraste
plt.figure()
plt.plot(grille, res, 'b-', marker='o', color='red')
plt.title('Evolution du contraste')
plt.xlabel('Grille')
plt.ylabel('Rapport max/min des normes')
plt.show()

# Tracer avec ylim
plt.figure()
plt.plot(grille, res, 'b-', marker='o', color='red')
plt.title('Evolution du contraste')
plt.xlabel('Grille')
plt.ylabel('Rapport max/min des normes')
plt.ylim(0, 3)
plt.axhline(y=1, color='black', linestyle='--')
plt.show()
