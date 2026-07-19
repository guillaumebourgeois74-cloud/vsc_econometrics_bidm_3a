
# %%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from stargazer.stargazer import Stargazer
import webbrowser
import numpy as np

# pour le test de Jarque Bera
from statsmodels.stats.stattools import jarque_bera

# Pour le test de Breusch-Pagan
from statsmodels.stats.diagnostic import het_breuschpagan

# Pour le test de white
from statsmodels.stats.diagnostic import het_white

# arrondis avec 2 décimales
pd.set_option("display.float_format", "{:.2f}".format)

def display(x):
    print(f"{x:.2f}")


# %%

# Création d'un petit jeu de données
df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 6, 8, 9]
})
print(df)

# %%
# Importation d'un "vrai" jeu de données
df = pd.read_csv(
    "C:/Users/julio/Downloads/brevets_depenses_RD_VIERGE.csv",
    sep=";",
    decimal=","
)


df.info() # informations sur le type de variable (les variables numériques ont-elles bien été comprises comme telle)

print(df.columns)

print(df.head())

#%%
#Renommer les variables 
df = df.rename(columns={
    "nb_brevets": "y",
    "depenses_RD": "x"
})
print(df.columns)

y = df["y"]
X = sm.add_constant(df["x"])

# %%
# Statistiques descriptives
desc = df[["x", "y"]].describe()

print(desc)

# %%
# Estimation de la régression

model = sm.OLS(y, X).fit()

#print(model.summary())

print(model.summary().tables[1])

#%%
# Tableau de décomposition de la variance et calcul du R²
# Valeurs prédites
y_hat = model.predict()
# Moyenne de Y
y_bar = df["y"].mean()
# Somme des carrés totale
SCT = ((df["y"] - y_bar)**2).sum()

# Somme des carrés expliquée
SCR = ((y_hat - y_bar)**2).sum()

# Somme des carrés des erreurs (résidus)
SCE = ((df["y"] - y_hat)**2).sum()

table_variance = pd.DataFrame({
    "Source": [
        "Régression",
        "Résidus",
        "Total"
    ],

       "Indicateur": [
        "SCR",
        "SCE",
        "SCT"
    ],
    "Somme des carrés": [
        SCR,
        SCE,
        SCT
    ]
})

table_variance

#%%
R2 = SCR / SCT

print("R² ")
display(R2)

# %%
# Sortie Stargazer

stargazer = Stargazer([model])

stargazer.title("Résultats de la régression")
stargazer.custom_columns(["Modèle 1"], [1])
stargazer.significance_levels([0.1, 0.05, 0.01])

#print(stargazer.render_html())
with open("regression.html", "w", encoding="utf-8") as f:
    f.write(stargazer.render_html())

webbrowser.open("regression.html")

# %%
# scatter plot
plt.scatter(df["x"], df["y"])
plt.plot(df["x"], model.predict(),
         color="red",
         linestyle=":",
         linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Régression linéaire simple")
plt.show()


# Calcul des coefficients à la main

# %%
# Covariance entre X et Y
cov_xy = np.cov(df["x"], df["y"], ddof=1)[0,1]
# Variance de X
var_x = np.var(df["x"], ddof=1)

# Coefficient beta1 chapeau
beta1_hat = cov_xy / var_x

print("Covariance X,Y :", cov_xy)
print("Variance X :", var_x)
print("Beta1 chapeau :", beta1_hat)

# Création du tableau de résultats
results = pd.DataFrame({
    "Statistique": [
        "Variance de X",
        "Covariance X,Y",
        "Beta1 chapeau"
    ],
    "Valeur": [
        var_x,
        cov_xy,
        beta1_hat
    ]
})

print(results)

# Enregistrement en CSV dans ton dossier courant
results.to_csv("C:/Users/julio/Desktop/resultats_regression.csv",
               sep=";",
               decimal=",",
               encoding="utf-8-sig",
                index=False)

#print("Fichier enregistré !")



# %%
# Analyse des résidus
residus = model.resid

print(residus)
print("Sommes des résidus", sum(residus))
df["residus"] = model.resid

df.head()

# Scatter plot des résidus
plt.scatter(df["x"], 
            df["residus"], 
            marker="s", # modifie le type de points 
            s=10, # Modifie la taille des points
            color="purple"
            )

plt.axhline(y=0, color="red",
              linestyle="--")

plt.xlabel("X")
plt.ylabel("Résidus de la régression")
plt.title("Nuage des résidus")

plt.show()



# %%
# Test de normalité des résidus Jarque Bera
residus = model.resid

# Test de Jarque-Bera
jb_stat, p_value, skewness, kurtosis= jarque_bera(residus)
print("H0: les résidus suivent une loi normale")
print("Statistique JB :", round(jb_stat,2))
print("p-value :", round(p_value,2))
print("Skewness :", round(skewness,2))
print("Kurtosis :", round(kurtosis,2))

# %%
# Tests d heteroscédasticité des résidus
# Test de Breusch-Pagan
bp_test = het_breuschpagan(model.resid, X)

labels = [
    "LM statistic",
    "LM p-value",
    "F statistic",
    "F p-value"
]
print("Rappel :\nH0 = Homoscédasticité\nvs\nH1 = Hétéroscédasticité")
print("\n--- Test de Breusch-Pagan ---")
for label, value in zip(labels, bp_test):
    print(label, ":", value)



# Test de White
white_test = het_white(model.resid, X)

labels = [
    "LM statistic",
    "LM p-value",
    "F statistic",
    "F p-value"
]
print("\n--- Test de White ---")
for label, value in zip(labels, white_test):
    print(label, ":", value)
# %%
# Test d'endogénéité et de corrélation sérielle
# Pour plus tard!
