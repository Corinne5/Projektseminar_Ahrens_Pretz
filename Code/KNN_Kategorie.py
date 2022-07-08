# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:22:01 2022

@author: pretz & ahrens


#-----------------------------------------------------------------------------#
#        Projektseminar Business Analytics - Sommersemester 2022              #
#-----------------------------------------------------------------------------#
#                      Künstliches neuronales Netz                            #
#      zur Vorhersage der Unfallkategorie zu einem bestimmten Zeitpunkt.      #      
#-----------------------------------------------------------------------------#
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import sys

start = time.time()

def read_data(stadt,wetter):
    '''
    Einlesen der Daten. Es werden sowohl die Trainings-Daten, als auch die 
    Daten, mit denen das neuronale Netz hinterher getestet werden soll 
    eingelesen.

    Returns
    -------
    x_train : DataFrame
        Trainingsdaten für das KNN.
        Beinhaltet Jahr, Monat, Tag, Stunde, durchschnittliche Temperatur, 
        durchschnittlicher Niederschlag zur entsprechenden Stunde
    y_train : DataFrame
        Trainingsadten für das KNN.
        Beinhaltet das entsprechende label zu den Daten aus x_train.
        (Anzahl der Unfälle mit Personenschaden zu angegebenem Zeitpunkt)
    x_test : DataFrame
        Testdaten für das trainierte KNN.
        Beinhaltet Jahr, Monat, Tag, Stunde, 
        durchschnittliche Temperatur, durchschnittlicher Niederschlag zur 
        entsprechenden Stunde
    y_test : DataFrame
        Testdaten für das trainierte KNN.
        Beinhaltet das entsprechende label zu den Daten aus x_test.
        (Anzahl der Unfälle mit Personenschaden zu angegebenem Zeitpunkt)
    x_pred : DataFrame
        Vorhersage für das trainierte KNN.
        Beinhaltet Jahr, Monat, Tag, Stunde, 
        durchschnittliche Temperatur, durchschnittlicher Niederschlag zur 
        entsprechenden Stunde.
        Auf Basis dieser Daten soll das KNN eine Voraussage für das Jahr x treffen.
    df_data : DataFrame
        Enhält alle Unfalldaten der Jahre 2016 bis 2020 inklusive der 
        Lufttemperaturwerte und der Niederschlagswerte.
        
    '''
    path_xtrain = "../Daten/Output/x_train_" + str(stadt) + "_cat.csv"
    x_train = pd.read_csv(path_xtrain, header = None)
    if wetter == "Nein":
        x_train = x_train.values[:,:4] # für Training ohne Wetterdaten
    
    path_ytrain ="../Daten/Output/y_train_" + str(stadt) + "_cat.csv"
    y_train = pd.read_csv(path_ytrain, header = None)
    
    path_xtest ="../Daten/Output/x_test_" + str(stadt) + "_cat.csv"
    x_test = pd.read_csv(path_xtest, header = None)
    if wetter == "Nein":
        x_test = x_test.values[:,:4] # für Training ohne Wetterdaten
    
    path_ytest ="../Daten/Output/y_test_" + str(stadt) + "_cat.csv"
    y_test = pd.read_csv(path_ytest, header = None)
    
    x_pred = pd.read_csv("../Daten/Output/x_pred.csv", header = None)
    
    path="../Daten/Output/Trainingsdaten_cat_" +str(stadt) +".csv"
    df_data = pd.read_csv(path)
    
    return x_train, y_train, x_test, y_test, x_pred, df_data


def weight_data(data):
    """
    Berechnet die Gewichtung der einzelnen Klassen.
    Klasse 0 Unfälle bis 5 Unfälle zu einem Zeitpunkt.
    
    Parameters
    ----------
    data : DataFrame
        Enhält alle Unfalldaten der Jahre 2016 bis 2020 inklusive der 
        Lufttemperaturwerte und der Niederschlagswerte.

    Returns
    -------
    class_weights : dict
        Enhält die Gewichtung der Klassen des Outputs.

    """
    #df_data.describe()

    # gibt an wie viele Daten von einer Klasse vorhanden sind
    num_class = np.bincount(data['category'])
    
    print(f"{num_class} gibt die Menge der Unfallanzahlen an")
    
    total = sum(num_class) # 10080
    
    Anteile = [num_class_i/total for num_class_i in num_class]
         
    for i,anteil in enumerate(Anteile):
        print(f" Zu allen stündliche Zeitpunkten von 2016 bis 2020 gabe es {100*anteil:.2f} % Unfälle mit {i} Personenschaden.")
              
    # berechen der Gewichtung jeder einzelnen Klasse
    weights = [(1/num_class_i)*(total/num_class.shape[0]) for num_class_i in num_class]
    
    class_weights = dict([(i, w) for i, w in enumerate(weights)])
    
    print("Die Gewichte der Klassen sind:")
    
    for i,c_weight in enumerate(class_weights):
        print(f'{c_weight:.2f} für Label {i}')
    
    return class_weights


def model_konf():
    '''
    Konfiguration des Tensorflow-Modells.
    Hier wird das künstliche neuronale Netz 
    erstellt, seine Layer und deren Aktivierungsfunktionen definiert und der Optimizer 
    (adam) festgelegt. 

    Returns
    -------
    model : keras.engine.sequential.Sequential
        Das konfigurierte Modell.

    '''
    # Dimension des Inputs (Jahr, Monat, Tag, Stunde) und somit Anzahl der künstlichen Neuronen im Inputlayer.
    Nin=np.shape(x_train)[1]

    # model definieren
    model = tf.keras.Sequential()

    # model anpassen
    # Definieren der Anzahl der Layer, Anzahl der Neuronen im Layer, Aktivierungsfunktionen
    model.add(tf.keras.layers.InputLayer(input_shape=Nin))
    #model.add(tf.keras.layers.Dense(100, activation='relu'))
    #model.add(tf.keras.layers.Dense(75, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    #model.add(tf.keras.layers.Dense(15, activation='relu'))
    model.add(tf.keras.layers.Dense(y_train.max()+1, activation='softmax'))
    
    # Modell kompilieren
    model.compile(optimizer= 'adam' ,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # f1 score ausprobieren

    # Modell als Übersicht ausgeben
    model.summary()

    return model


def model_train(model, x, y): # class_weights
    '''
    Trainieren des vorher konfigurierten Modell.

    Parameters
    ----------
    model : keras.engine.sequential.Sequential
        Das konfigurierte Modell.
    x : Array
        Inputdaten für das KNN.
    y : DataFrame
        Entsprechende Outputdaten.

    Returns
    -------
    history: keras.callbacks.History
        Speichert den Loss und die Accuracy ab.
        Sowohl für Trainingsdaten als auch für die Validierungdaten.

    '''
    # Verlauf des Trainings, speichert unter anderem loss, accuracy Werte 
    history = model.fit(x, y, epochs=200, batch_size=50,verbose=1, validation_split = 0.1)#, class_weight=class_weight)

    return history

def save_model(model):
    '''
    Abspeichern des Trainierten Modells zum späteren einlesen.

    Parameters
    ----------
    model : keras.engine.sequential.Sequential
        Das konfigurierte Modell.

    Returns
    -------
    None.

    '''
    if wetter == 'Ja':
        path_model = "../Daten/Output/model_cat_" + str(stadt) +"_weather"
    else:
        path_model = "../Daten/Output/model_cat_" + str(stadt)
    model.save(path_model)
    return


def model_test(model, x_test, y_test):
    '''
    Testen des trainierten KNNs anhand der Test-Daten

    Parameters
    ----------
    model : keras.engine.sequential.Sequential
        Das konfigurierte Modell.
    x_test : DataFrame
        Testdaten für das trainierte KNN.
        Beinhaltet Jahr, Monat, Tag, Stunde, 
        durchschnittliche Temperatur, durchschnittlicher Niederschlag zur entsprechenden Stunde.
    y_test : DataFrame
       Testdaten für das trainierte KNN.
       Beinhaltet das entsprechende label zu den Daten aus x_test.
       (Anzahl der Unfälle mit Personenschaden zu angegebenem Zeitpunkt).
       
    Returns
    -------

    predictions_x_test : DataFrame
        Vorhersage für den Output auf Grundlage der Daten von x_test.
        
    predictions_with_input : DataFrame
        Vorhersage mit Inputdaten zusammengeführt.

    '''

    
    predictions_x_test = pd.DataFrame(model.predict(x_test)) 
    
    predictions_with_input = pd.DataFrame(np.c_[x_test, predictions_x_test])
    
    return predictions_x_test, predictions_with_input


def data_hist(y_test, predictions):
    '''
    Daten zum Vergleich der Vorhersagewerte und den tatsächlichen Werten.

    Parameters
    ----------
    y_test : DataFrame
        Tatsächliche Werte.
    predictions : DataFrame
        Vorhergesagte Werte.

    Returns
    -------
    diff : array
        Differenzen zwischen der Vorhersage und den tatsächlichen Werten.

    '''
    predictions = np.array(predictions)
    y_test = y_test.to_numpy()
    diff = [(np.argmax(predictions[i]))-y_test[i] for i in range(len(y_test))]
    
    diff = np.array(diff)
    
    unique, counts = np.unique(diff, return_counts=True)
    
    for u, c in zip(unique, counts):
        print(f'# {u} : {c}')
        
    return diff

'''
Darstellung der Ergebnisse
'''
#print(history_cat.history.keys())

def plots(history):
    
    # Accuracy und Loss
    fig1, ax1 = plt.subplots(2,1,sharex=True, figsize=(17, 10), dpi=500)
    ax1[0].grid()
    ax1[1].grid()
    plt.xlabel('Epochen', fontsize=18)
    ax1[0].tick_params(axis='both', which='major', labelsize=18)
    ax1[0].tick_params(axis='both', which='minor', labelsize=18)
    ax1[1].tick_params(axis='both', which='major', labelsize=18)
    ax1[1].tick_params(axis='both', which='minor', labelsize=18)
    ax1[0].plot(history.history["loss"], linewidth=3)
    ax1[0].set_ylabel("Loss", fontsize=18)
    ax1[1].plot(history.history["accuracy"], linewidth=3)
    ax1[1].set_ylabel("Accuracy", fontsize=18)
    
    #hist
    fig2, ax2 = plt.subplots(1,1,sharex=True, figsize=(17, 10), dpi=300)
    ax2.set_xticks([-5,-4, -3, -2, -1, 0, 1,2,3,4])
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='minor', labelsize=18)
    plt.xlabel('Differenz zur tatsächlichen Unfallanzahl', fontsize=18)
    plt.ylabel('Anzahl', fontsize=18)
    ax2.hist(diff, bins=[-5,-4.5,-4,-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4,4.5])
    plt.show()
    
    #val acc
    fig3, ax3 = plt.subplots(2,1,sharex=True, figsize=(17, 10), dpi=500)
    ax3[0].grid()
    ax3[1].grid()
    plt.xlabel('Epochen', fontsize=18)
    ax3[0].tick_params(axis='both', which='major', labelsize=18)
    ax3[0].tick_params(axis='both', which='minor', labelsize=18)
    ax3[1].tick_params(axis='both', which='major', labelsize=18)
    ax3[1].tick_params(axis='both', which='minor', labelsize=18)
    ax3[0].plot(history.history["val_loss"], linewidth=3)
    ax3[0].set_ylabel("validation Loss", fontsize=18)
    ax3[1].plot(history.history["val_accuracy"], linewidth=3)
    ax3[1].set_ylabel("validation Accuracy", fontsize=18)
    
    return


#Einlesen der Daten
print("Auf Grundlage welcher Stadtdaten soll das KNN trainiert werden? Geben Sie eine Stadt ein:")
stadt = input()

if stadt not in ['Mainz','Wiesbaden']:
    print("Für diese Stadt liegen leider keine Daten vor.")
    sys.exit()
    
print("Sollen Wetterdaten (Temperatur und Niederschlag) in das Training mit einbezogen werden? Antworten sie Ja oder Nein.")
wetter = input()
    
#Einlesen der Daten
x_train, y_train, x_test, y_test, x_pred, df_data = read_data(stadt, wetter) 

#Gewichte der Klassen berechnen
class_weights = weight_data(df_data)

#Konfigurieren des Modells
model_cat = model_konf()

# Trainieren des Modells
history_cat = model_train(model_cat, x_train, y_train)#, class_weights)

#Abspeichern des trainierten Modells
save_model(model_cat)

#Testen des trainierten Modells
predictions_y, predictions_with_input = model_test(model_cat, x_test, y_test)

#Vergleichen der Vorhersage zu den tatsächlichen Wertem
diff = data_hist(y_test, predictions_y)

#Plotten der Ergebnisse
plots(history_cat)

end = time.time()
print(f'time : {end - start}')