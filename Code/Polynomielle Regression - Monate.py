### Polynomielle Regression - Mainz (Monate) ###

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import f1_score

# Modell
def poly_reg_months():
    '''
    

    Returns
    -------
    None.

    '''
    # Daten einlesen
    df = pd.read_csv("Unfallorte2016.csv", sep = ";")
    
    df = df.loc[df["ULAND"] == 7]
    df = df.loc[df["UREGBEZ"] == 3]
    df = df.loc[df["UKREIS"] == 15]
    df = df.loc[df["UGEMEINDE"] == 0]
    
    # Zeitreihe erstellen
    df = df.rename(columns = {"UJAHR": "Jahr"})
    df = df.rename(columns = {"UMONAT": "Monat"})
    df = df.rename(columns = {"UKATEGORIE": "Anzahl Unfälle"})
    df = df.groupby(["Jahr", "Monat"]).count()
    df = df.reset_index()
    df = df[["Monat", "Anzahl Unfälle"]]
    print(df)
    
    # Daten aufbereiten
    x = np.array(df["Monat"]).reshape(-1, 1)
    plt.xlabel("Monat")
    y = np.array(df["Anzahl Unfälle"]).reshape(-1, 1)
    plt.ylabel("Anzahl Unfälle")
    plt.plot(y, "-m")
    #plt.show()
    
    # Daten quadrieren
    polyFeat = PolynomialFeatures(degree = 5)
    x = polyFeat.fit_transform(x)
    
    # Daten trainieren
    model = linear_model.LinearRegression()
    model.fit(x, y)
    accuracy = model.score(x, y)
    print("Genauigkeit:", round(accuracy * 100, 3), "%")
    
    # Trainierte Daten plotten
    y_pred = model.predict(x)
    plt.plot(y_pred, "--b")
    plt.show()
    
    # F1 Score
    y_pred_new = np.argmax(model.predict(x), axis=-1)
    f1_macro = f1_score(y, y_pred_new, average = "macro")
    print("Macro F1_Score:", round(f1_macro * 100, 3), "%")
    f1_micro = f1_score(y, y_pred_new, average = "micro")
    print("Micro F1_Score:", round(f1_micro * 100, 3), "%")
    f1_weighted = f1_score(y, y_pred_new, average = "weighted")
    print("Weighted F1_Score:", round(f1_weighted * 100, 3), "%")
    
    # Vorhersage
    months = int(input("Bitte geben Sie den Zeitraum der Vorhersage an: "))
    print("Vorhersage nach", months, "Monaten: ", end = "")
    pred = round(int(model.predict(polyFeat.fit_transform([[df.shape[0] + months]]))), 2)
    print(pred, "Unfälle")
    
    # Vorhersage plotten
    x1 = np.array(list(range(1, df.shape[0] + months))).reshape(-1, 1)
    y1 = model.predict(polyFeat.transform(x1))
    plt.plot(y1, "--r")
    plt.plot(y_pred, "--b")
    plt.show()

# Ploteinstellungen
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Ausgabe
poly_reg_months()