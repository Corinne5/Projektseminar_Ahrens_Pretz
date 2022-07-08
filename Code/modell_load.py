# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:18:35 2022

@author: pretz & ahrens

#-----------------------------------------------------------------------------#
#        Projektseminar Business Analytics - Sommersemester 2022              #
#-----------------------------------------------------------------------------#
#                  Test des künstlichen neuronalen Netzes                     #
#   Nutzung eines vergleichbaren Datensatzes mit dem das KNN traniert wurde   #  
#            zusätzlich kann ein bestimmtes Datum eingegeben werden           #
#-----------------------------------------------------------------------------#
"""

import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
from sklearn.metrics import f1_score


def read_data(stadt, wetter, column, typ):
    '''
    Einlesen der Daten

    Parameters
    ----------
    stadt : str
        Stadt für die eine Vorhersage getroffen werden soll.
    wetter : str
        Gibt ein ob Wetterdaten miteinbezogen werden sollen oder nicht.
    column : str
        Gibt an welche Spalte der Daten die wahre Vorhersage ist.
    typ : str
        Gibt an ob die Kategorie oder die Unfallanzahl vorhergesagt werden soll.

    Returns
    -------
    x : DataFrame
        Enthält die InputDaten, auf Grundlage derer eine Voraussage getroffen wird.
    y : array
        Enthält die tatsächlichen Werte zum späteren Vergleich zu Vorhersage.

    '''
    
    path="../Daten/Output/" + 'Trainingsdaten_' + str(typ) + '_' + str(stadt) +".csv"
    df_data = pd.read_csv(path)
            
    y = df_data.values[:,4:5]
    x = df_data.drop(columns = [column])
    if wetter == "Nein":
        x = x.values[:,:4]


    return x, y

def load_model(wetter, model_typ, stadt_knn):
    '''
    Einlesen und laden des trainierten Modells.

    Parameters
    ----------
    wetter : str
        Gibt an ob das Modell geladen werden soll, was mit Wetterdaten 
        trainiert wurde oder das ohne.
    model_typ : str
        Gibt an welches Modell geladen werden soll hinsichtlich der Vorhersage.
        (Unfallanzahl oder Kategorie)
    stadt_knn : str
        Gibt an welches Modell bezüglich der Stadt geladen werden soll.
        wenn stadt = 'Mainz' dann 'Wiesbaden'
        wenn stadt = 'Wiesbaden' dann 'Mainz'

    Returns
    -------
    model : keras.engine.sequential.Sequential
        Traniertes Modell.

    '''
    
    if wetter == 'Nein':
        model_path = '../Daten/Output/model_' + str(model_typ) + '_' + str(stadt_knn)
    else:
        model_path = '../Daten/Output/model_' + str(model_typ) + '_' + str(stadt_knn) + '_weather'
        
    model = keras.models.load_model(model_path)
    
    return model

 
def model_test(model, x):
    '''
    Testen des trainierten KNNs anhand der Daten x.

    Parameters
    ----------
    model : keras.engine.sequential.Sequential
        Das trainierte Modell.
    x : DataFrame
        Testdaten für das trainierte KNN.
       
    Returns
    -------

    prediction : DataFrame
        Vorhersage für den Output auf Grundlage der Daten von x.
        
    predictions_with_input : DataFrame
        Vorhersage mit Inputdaten zusammengeführt.

    '''

    
    prediction = pd.DataFrame(model.predict(x)) 
    
    predictions_with_input = pd.DataFrame(np.c_[x, prediction])
    
    return prediction, predictions_with_input


def data_hist(y, predictions):
    '''
    Daten zum Vergleich der Vorhersagewerte und den tatsächlichen Werten.

    Parameters
    ----------
    y : DataFrame
        Tatsächliche Werte.
    predictions : DataFrame
        Vorhergesagte Werte.

    Returns
    -------
    diff : array
        Differenzen zwischen der Vorhersage und den tatsächlichen Werten.

    '''
    predictions = np.array(predictions)
    
    diff = [(np.argmax(predictions[i]))-y[i] for i in range(len(y))]
    
    diff = np.array(diff)
    
    
    unique, counts = np.unique(diff, return_counts=True)
    result_dict = dict(zip(unique, counts))
        
    for u, c in zip(unique, counts):
        print(f'# {u} : {c}')
    print('------------------------------------------------')   
    correct = result_dict.get(0.0, 0)
    total = np.sum(counts)
        
    print(f'Es wird eine Accuracy von {correct/total:.2f} erreicht')
    print('------------------------------------------------')
    return diff


def histogram(diff, stadt, stadt_knn):
    '''
    Histogramm zur veranschulichung der Differenz zwischen 
    tatsächlichem und vorhergesagtem Wert

    Parameters
    ----------
    diff : array
        Enthält die Differenzen von Vorhersage und tatsächlichem Wert.

    Returns
    -------
    None.

    '''
    
    fig2, ax2 = plt.subplots(1,1,sharex=True, figsize=(17, 10), dpi=300)
    ax2.set_xticks([-5,-4, -3, -2, -1, 0, 1,2,3,4])
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='minor', labelsize=18)
    plt.title(f'Vorhersage für {stadt} auf Grundlage des KNN {stadt_knn}',fontsize=20)
    plt.xlabel('Differenz zur tatsächlichen Unfallanzahl', fontsize=18)
    plt.ylabel('Anzahl', fontsize=18)
    ax2.hist(diff, bins=[-5,-4.5,-4,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4,4.5])
    plt.show()
    
    return

def f1(y_test, predictions):
    
    max_unfaelle = int(y_test.max())
    predictions = np.array(predictions)
    
    f1_macro = f1_score(y_test, np.argmax(predictions, axis = -1), average = 'macro')
    print(f'f1_macro : {f1_macro}')
    print('------------------------------------------------')
    f1_label = [f1_score(y_test, np.argmax(predictions, axis = -1), labels= [i], pos_label=1, average='macro') for i in range(0,max_unfaelle+1)]
    
    for label,f1 in enumerate(f1_label):
        print(f'Der f1-Score der Klasse {label} beträgt: {f1}.')


    return f1_macro, f1_label





print('Soll die Unfallanzahl oder die Kategorie des Unfalls vorhergesagt werden? Geben Sie U für die Unfallanzahl und K für die Kategorie an:')
typ = input()
if typ == 'U':
    column = '# Unfaelle'
    model_typ = 'num'
    typ = 'AnzahlUnfaelle'
else:
    column = 'category'
    model_typ = 'cat'
    typ = 'cat'
    
    
print('Möchten Sie ein genaues Datum eingeben? Antworten Sie Ja oder Nein:')
date = input()

if date == 'Ja':
    print('In welche Stadt möchten Sie fahren ? Mainz oder Wiesbaden?')
    stadt = input()
    if stadt == 'Mainz':
        stadt_knn = 'Mainz'
    else:
        stadt_knn = 'Wiesbaden'
    
    print('Möchten Sie zusätzlich zum Zeitpunkt auch Wetterdaten mit eingeben? Antworten Sie Ja oder Nein:')
    wetter = input()
    
    print('Geben Sie ein Jahr ein:')
    year = int(input())
    print('Geben Sie einen Monat ein (1-12):')
    month = int(input())
    print('Geben Sie ein Wochentag als Zahl ein (0 = Sonntag):')
    day = int(input())
    print('Geben Sie eine Stunde ein (1-24):')
    hour = int(input())
    
    if wetter == 'Ja':
        print('Wie viel Niederschlag wird erwartet?')
        nied = float(input())
        print('Wie hoch ist die Lufttempreatur?')
        temp = float(input())
        x = np.array([year, month, day, hour, nied, temp])
    else:
        x = np.array([year, month, day, hour])
    
    x = np.reshape(x, (1,len(x)))
    
    model = load_model(wetter, model_typ, stadt_knn)
    prediction, prediction_with_input = model_test(model,x)
    print(prediction_with_input)


else:    
    print('Wollen sie das KNN mit Wetterdaten testen? Geben sie Ja oder Nein ein:')
    wetter = input()
    print('Für welche Stadt soll das KNN eine Vorhersage machen?')
    stadt = input()
    
    
    if stadt == 'Mainz':
        stadt_knn = 'Wiesbaden'
    else: 
        stadt_knn = 'Mainz'
        
    x,y = read_data(stadt,wetter,column,typ)
    model = load_model(wetter, model_typ, stadt_knn)
    prediction, prediction_with_input = model_test(model,x)
    diff = data_hist(y, prediction)
    histogram(diff, stadt, stadt_knn)
    f1_macro, f1_label = f1(y,prediction)
        

