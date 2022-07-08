### Polynomielle Regression - Mainz ###

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import f1_score

# Anzeigeoptionen des DataFrames
pd.set_option("display.max_rows", 10)

# Modell
def poly_reg_days():
    '''
    

    Returns
    -------
    None.

    '''
    # Daten einlesen
    df = pd.read_csv("../Daten/Output/Unfall_Trainingsdaten.csv", sep = ",")
    
    # Spalten umbenennen
    df = df.rename(columns = {"0": "Jahr"})
    df = df.rename(columns = {"1": "Monat"})
    df = df.rename(columns = {"2": "Tag"})
    df = df.rename(columns = {"3": "Stunde"})
    df = df.rename(columns = {"4": "Temperatur"})
    df = df.rename(columns = {"5": "Niederschlag"})
    df = df.rename(columns = {"6": "Anzahl Unfälle"})
    
    # Zeitreihe erstellen
    df = df.groupby(["Jahr", "Monat", "Tag"])["Anzahl Unfälle"].sum()
    df = df.reset_index()
    df = df.rename(columns={"index": "Tag"})
    df["Tag"] = df.index + 1
    df = df[["Tag", "Anzahl Unfälle"]]
    print(df)
    
    # Daten aufbereiten
    x = np.array(df["Tag"]).reshape(-1, 1)
    plt.xlabel("Tag")
    y = np.array(df["Anzahl Unfälle"]).reshape(-1, 1)
    plt.ylabel("Anzahl Unfälle")
    plt.plot(y, "-m")
    #plt.show()
    
    # Daten quadrieren
    polyFeat = PolynomialFeatures(degree = 6)
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
    days = int(input("Bitte geben Sie den Zeitraum der Vorhersage an: "))
    print("Vorhersage nach", days, "Tagen: ", end = "")
    pred = round(int(model.predict(polyFeat.fit_transform([[df.shape[0] + days]]))), 2)
    print(pred, "Unfälle")
    
    # Vorhersage plotten
    x1 = np.array(list(range(1, df.shape[0] + days))).reshape(-1, 1)
    y1 = model.predict(polyFeat.transform(x1))
    plt.plot(y1, "--r")
    plt.plot(y_pred, "--b")
    plt.xlabel("Tag")
    plt.ylabel("Anzahl Unfälle")
    plt.show()

# Ploteinstellungen
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Ausgabe
poly_reg_days()