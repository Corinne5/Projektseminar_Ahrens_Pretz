# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:20:48 2022

@author: pretz & ahrens

#-----------------------------------------------------------------------------#
#        Projektseminar Business Analytics - Sommersemester 2022              #
#-----------------------------------------------------------------------------#
#                         Datenaufbereitung                                   #
#            Daten sollen zum Training eines KNN genutzt werden               #      
#-----------------------------------------------------------------------------#
"""
import pandas as pd
import numpy as np
import numpy.matlib as mnp
import datetime
from sklearn.model_selection import train_test_split
import sys

# Konvertieren einer pdf Tabelle zu einer csv Datei
#tabula.convert_into(r'C:\Users\pretz\SynologyDrive\UNI_SIEGEN\Master\
    #2. Semester\Projektseminar\SOSE22\AGS.pdf',r'C:\Users\pretz\SynologyDrive
    #\UNI_SIEGEN\Master\2. Semester\Projektseminar\SOSE22\AGS.csv', 
    #output_format="csv", pages="all")

def einlesenDaten(years,stadt):
    '''
    Einlesen aller benötigten Daten
    AGS - Nummern aller Deutschen Städte/Gemeinden.
    Unfallatlas Deutschland alle vorhandenen Jahr.
    Stündliche Niederschlags und Temperatur Werte aller vorhandenen Jahr in Deutschland.
    
    Parameters
    ----------
    years : list
        Enthält alle Jahre, für die Unfalldaten vorliegen.

    Returns
    -------
    ags_gemeinden : DataFrame
        Enthält alle AGS-Nummern und die entsprechenden Namen der 
        Städte/Gemeinden aus Deutschland.
    unfaelle_de : list
        Enhält DataFrames die jeweils alle Unfaelle mit Personenschaden und 
        dazugehörige Informationen aus einem Jahr aus years enthalten.
    data_niederschlag : DataFrame
        Enthält die stündlichen Niderschlagswerte der Stadt Mainz.
    data_temp : DataFrame
        Enthält die stündlichen Lufttemperaturwerte der Stadt Mainz.

    '''
    
    ags_gemeinden = pd.read_csv("../Daten/Input/AGS.csv", sep = ",", encoding= 'unicode_escape')
    path_nied = "../Daten/Input/Niederschlagshöhe_" + str(stadt) + ".csv"
    data_niederschlag = pd.read_csv(path_nied, sep=";")
    path_temp = "../Daten/Input/Temperatur_" + str(stadt) + ".csv"
    data_temp = pd.read_csv(path_temp, sep=";")
    data_temp = data_temp[data_temp.TT_TU != -999]

    unfaelle_de = []
    
    for i in years:
        
        dateiname = '../Daten/Input/Unfallorte' + str(i) + '.csv'
        
        unfaelle_de.append(pd.read_csv(dateiname, sep= ";") )
    
    return ags_gemeinden, unfaelle_de, data_niederschlag, data_temp


def unfaelle_re (ags_gemeinden, unfaelle_de, stadt, jahr):
    '''
    Reduzieren der DataFrames in unfaelle_de auf die gewünschte Stadt/Gemeinde.

    Parameters
    ----------
    ags_gemeinden : DataFrame
        Enhält alle AGS-Nummern deutscher Städte und Gemeinden.
    unfaelle_de : list
        Enhält DataFrames die jeweils alle Unfaelle mit Personenschaden und 
        dazugehörige Informationen aus einem Jahr aus years enthalten.
        Ganz Deutschland.
    stadt : str
        Stadt/Gemeinde auf die der DataFrame reduiziert werden soll.
    jahr : int
        Jahr für das das DataFrame reduziert werden soll.

    Returns
    -------
    unfaelle_region : list
        Enhält DataFrames die jeweils alle Unfaelle mit Personenschaden und 
        dazugehörige Informationen aus einem Jahr aus years für eine Stadt/Gemeinde enthalten.

    '''
    
    # Suche Informationen der eingegebenen Stadt/Gemeinde in ags_gemeinden.
    info_stadt = ags_gemeinden[(ags_gemeinden['Stadt / Gemeinde'] == stadt)] 
    
    # Gemeindeschlüssel aus info_stadt als String aufteilen und einzeln in int umwandeln
    ags_str = str(info_stadt['Gemeindeschlüssel'].values[0])
    land = int(ags_str[0])          # 1. Stelle -> Nummer des Landes
    bezirk = int(ags_str[1])        # 2. Stelle -> Nummer des Bezirks
    kreis = int(ags_str[2:4])       # 3. bis 4. Stelle -> Nummer des Kreises
    gemeinde = int(ags_str[4:7])    # 5. bis 7. Stelle -> Nummer der Gemeinde

    # Suche in unfaelle_de mit der vorher ermittelten AGS - Nummer die 
    # Unfalleinträge
    unfaelle_region = unfaelle_de[(unfaelle_de['ULAND'] == land) & 
                                (unfaelle_de['UREGBEZ'] == bezirk) & 
                                (unfaelle_de['UKREIS'] == kreis) &
                                (unfaelle_de['UGEMEINDE'] == gemeinde)]
    
    # Anzahl der Spalten ist dabei die Anzahl der Unfälle in der gesuchten 
    # Stadt/Gemeinde.
    num_unfaelle = unfaelle_region.shape[0]
    
    print(f"In {stadt} gab es {jahr} {num_unfaelle} Unfälle")
    
    return unfaelle_region

def index_from_mdh(m, d, h):
    """
    Funktion, welche die Indizes des DataFrames unfaelle_region berechnet.

    Parameters
    ----------
    m : int
        Monat.
    d : int
        Tag.
    h : int
        Stunde.

    Returns
    -------
    idx : int
        Index

    """
    m_idx = (m-1) * 24 * 7
    d_idx = (d-1) * 24
    h_idx =  h-1
    idx = m_idx + d_idx + h_idx
    
    return idx

def prepare_data(unfaelle_region,jahr):
    '''
    
    Umändern der DataFrames in unfaelle_region zu zwei neuen DataFrames
    
    Neue DataFrames sollen als Trainingsdaten für ein KNN genutzt werden.
    
    1.
    Die DataFrames in unfaelle_region werden  um alle Zeitpunkte erweitert in denen kein 
    Unfall mit Personenschaden entstanden ist.
    Dabei wird zusätzlich gezählt wie viele Unfälle zu einem Zeitpunkt geschehen sind.
    
    2.
    Die DataFrames in unfaelle_region werden reduziert auf Jahr, Monat, Tag , Stunde, Kategorie.
    
    
    Parameters
    ----------
    unfaelle_region : list
        Enhält DataFrames die jeweils alle Unfaelle mit Personenschaden und 
        dazugehörige Informationen aus einem Jahr aus years für eine Stadt/Gemeinde enthalten.
    jahr : int
        Jahr für das der DataFrame angepasst werden soll.

    Returns
    -------
    u_matrix : list
        DataFrames beinhalten alle Zeitpunkte der Jahre in years und die Anzahl der zu diesem Zeitpunkt geschehenen Unfaelle.
    u_matrix_cat : list
        DataFrames behinhalten den Zeitpunkt und die Unfallkategorie des entstandenen Unfalls.

    '''
    
    ########################################################################
     
    # erstelle unfall matrix mit 0 Einträgen, 12Monate x 7Tage x 24Stunden = 2016
    # 4 Spalten für Monat, Tag, Stunde, Anzahl Unfälle
    u_matrix = np.zeros((2016,4))
    
    jahre = np.repeat(jahr,2016)
    # 1 bis 24 Stunden, 7 mal wiederholen, 12 mal wiederholen 
    # Stunden 1 bis 24 7*12 wiederholt -> 1 x 2016 Matrix
    stunden = mnp.repmat(np.arange(1,25,1), 1,12*7)
    
    # 1 bis 7 Tage, jeweils 24 mal, 12 mal wiederholt
    # 1 bis 7 jeweils 24 wiederholen, dann insgesamt 12 wiederholen -> 1 x 2016 Matrix
    tage = mnp.repmat(np.repeat(np.arange(1,8,1),24),1,12)
    
    # 1 bis 12 Monate, jeweils 24*7 mal
    # 1 bis 12 jeweils 24*7 wiederholt -> 2016 x 1 Matrix
    monat = np.repeat(np.arange(1,13,1), 24*7)
    
    # Null-Spalte für Anzahl der Unfälle -> 2016 x 1 Matrix
    u_zeros = np.zeros((2016,1))
    
    # Matrix zusammenbauen
    # Dazu tage und stunden zu 2016 x 1 Matrix transponieren
    u_matrix = np.c_[jahre, monat, tage.T, stunden.T, u_zeros]
    
    # Monate, Wochentage und Stunden durchgehen um Unfallspalte zu füllen
    # alle monat, tag, stunden kombinationen in unfaelle_region durchgehen
    # sobald Eintrag gefunden wird, wird die Unfallspalte um 1 erhöht
    for m, d, h in zip(unfaelle_region.UMONAT, unfaelle_region.UWOCHENTAG, unfaelle_region.USTUNDE):
        # Index in der Matrix ermitteln
        idx = index_from_mdh(m, d, h) 
        
        # In Zeile 'idx' die Spalte 'Anzahl an Unfälle' um 1 erhöhen
        u_matrix[idx, -1] += 1
    
    #######################################################################
    
    # Reduzieren des DataFrames unfaelle_region auf JAHR, MONAT, TAG, STUNDE, UKATEGORIE
    u_matrix_cat = unfaelle_region.drop(columns=['OBJECTID','ULAND',
                                                    'UREGBEZ','UKREIS',
                                                    'UGEMEINDE',
                                                    'UART','UTYP1',
                                                    'ULICHTVERH','IstRad',
                                                    'IstPKW','IstFuss',
                                                    'IstKrad','IstGkfz',
                                                    'IstSonstig','LINREFX',
                                                    'LINREFY','XGCSWGS84',
                                                    'YGCSWGS84','USTRZUSTAND'])#'UKATEGORIE',
    
    u_matrix_cat = u_matrix_cat.reindex(columns=['UJAHR','UMONAT','UWOCHENTAG','USTUNDE','UKATEGORIE'])
    u_matrix_cat.columns=['year','month','weekday','hour','category']
    
    return  u_matrix, u_matrix_cat



def merge_data(u_matrix, mean_vals_nied, mean_vals_temp, stadt, u_matrix_cat):
    '''
    Zusammenführen der Unfall und Wetterdaten.

    Parameters
    ----------
    u_matrix : list
        Enthält alle DataFrames der Jahre 2016 bis 2020, die Zeitpunkt und Unfallanzahl abgespeichert haben.
    mean_vals_nied : DataFrame
        Enhält die Durchschnitswerte des Niederschlags zu den Zeitpunkten aus u_matrix.
    mean_vals_temp : DataFrame
        Enhält die Durchschnitswerte der Temperatur zu den Zeitpunkten aus u_matrix.
    stadt : str
        Stadt für die die Daten gesucht werden.
    u_matrix_cat : list
        Enthält alle DataFrames der Jahre 2016 bis 2020, die Zeitpunkt und Unfallkategorie abgespeichert haben.

    Returns
    -------
    data_knn_cat : DataFrame
        Alle Zeitpunkte von 2016 bis 2020 in denen ein Unfall mit 
        Personenschaden entstande ist und die Kategorie des Unfalls.
    data_knn_num : DataFrame
        Alle Zeitpunkte von 2016 bis 2020, die Wetterdaten und die Anzahl 
        der Unfaelle mit Personenschaden.

    '''
    
    data_cat = pd.DataFrame(np.row_stack(u_matrix_cat), columns=['year', 'month', 'weekday', 'hour', 'category'])
    data_knn_cat = pd.merge(data_cat, mean_vals_nied,  how='left', left_on=['year', 'month', 'weekday', 'hour'], right_on=['year', 'month', 'weekday', 'hour'])
    data_knn_cat = pd.merge(data_knn_cat, mean_vals_temp,  how='left', left_on=['year', 'month', 'weekday', 'hour'], right_on=['year', 'month', 'weekday', 'hour'])
    
    
    data_num = pd.DataFrame(np.row_stack(u_matrix), columns=['year', 'month', 'weekday', 'hour', '# Unfaelle'])
    data_knn_num = pd.merge(data_num, mean_vals_nied,  how='left', left_on=['year', 'month', 'weekday', 'hour'], right_on=['year', 'month', 'weekday', 'hour'])
    data_knn_num = pd.merge(data_knn_num, mean_vals_temp,  how='left', left_on=['year', 'month', 'weekday', 'hour'], right_on=['year', 'month', 'weekday', 'hour'])
   
    
    #data_knn_num = pd.DataFrame(np.c_[data_num['year'],data_num['month'],data_num[2],data_num[3], mean_vals_temp['val'], mean_vals_nied['val'], data_num[4]])#, columns=['year', 'month', 'weekday', 'hour', '# Unfaelle','temp', 'nied'])
    

    dateiname = '../Daten/Output/Trainingsdaten_cat_' + str(stadt) + '.csv'
    data_knn_cat.to_csv(dateiname,index = False, sep = ',')
    
    dateiname = '../Daten/Output/Trainingsdaten_AnzahlUnfaelle_' + str(stadt) + '.csv'
    data_knn_num.to_csv(dateiname,index = False, sep = ',')
    
    return data_knn_cat, data_knn_num



def pred_data():
    '''
    Aufstellen des DataFrames zur Vorhersage für das Jahr 2021.

    Returns
    -------
    x_pred : DataFrame
        Enthält alle Zeitpunkte des Jahres 2021 und die entsprechenden Wetterdaten.
        Dieser Dataframe soll zu Vorhersage der Unfallanzahlen im Jahr 2021 genutzt werden

    '''

    x_pred = np.zeros((2016,4))

    jahre = np.repeat(2021,2016)
    # 1 bis 24 Stunden, 7 mal wiederholen, 12 mal wiederholen 
    # Stunden 1 bis 24 7*12 wiederholt -> 1 x 2016 Matrix
    stunden = mnp.repmat(np.arange(1,25,1), 1,12*7)

    # 1 bis 7 Tage, jeweils 24 mal, 12 mal wiederholt
    # 1 bis 7 jeweils 24 wiederholen, dann insgesamt 12 wiederholen -> 1 x 2016 Matrix
    tage = mnp.repmat(np.repeat(np.arange(1,8,1),24),1,12)

    # 1 bis 12 Monate, jeweils 24*7 mal
    # 1 bis 12 jeweils 24*7 wiederholt -> 2016 x 1 Matrix
    monat = np.repeat(np.arange(1,13,1), 24*7)

    # Matrix zusammenbauen
    # Dazu tage und stunden zu 2016 x 1 Matrix transponieren
    x_pred = pd.DataFrame(np.c_[jahre, monat, tage.T, stunden.T])

    x_pred.to_csv('../Daten/Output/x_pred.csv', sep=',')
    
    return x_pred

def average_data(data_frame, data_key, years,typ):
    '''
    Berechnung der durchschnittlichen Temperatur/Niederschlagswerte von 2016 bis 2021

    Parameters
    ----------
    data_frame : DataFrame
        Temperatur oder Niederschlagswerte des Deutschen Wetterdienstes über alle aufgezeichneten Jahre
    data_key : str
        Spaltenname der gewünschten Daten (Temperatur oder Niederschlagswert).
    years : list
        Liste der Jahre, für die die Daten benötigt werden.

    Returns
    -------
    mean_vals : DataFrame
        Enthält die Durchschnittswerte der Temperatur/Niederschlagswerte zu den gegebenen Zeitpunkten.

    '''
    # datum extrahieren
    mess_datum = list(data_frame['MESS_DATUM'])
    # Temperatur/ Niederschlag extrahieren
    vals =  list(data_frame[data_key])
    
    # in datetime umwandeln
    # dates = []
    # for date in mess_datum:
    #     dates.append(datetime.datetime.strptime(str(date), '%Y%m%d%H'))
    dates = [datetime.datetime.strptime(str(date), '%Y%m%d%H') for date in mess_datum]
    
    # neuer Dataframe, die Jahr, Monat, Wochentag und Stunde und Wert enthält
    df_ymwhw = pd.DataFrame([[d.year,d.month,d.weekday() + 1, d.hour + 1, v] for d, v in zip(dates,vals) if d.year in years], columns=['year', 'month', 'weekday', 'hour', 'val_'+str(typ)])
    
    mean_vals = df_ymwhw.groupby(['year', 'month', 'weekday', 'hour']).mean()
   
    return mean_vals

def data_knn(df_data, typ, column, stadt):
    
    """
    Aufteilen der Daten in Input (x) - und Outputdaten (y).
    Zusätzliche Aufteilung der Daten in Trainings und Testdaten.
    Als Testdaten werden 20 % der Gesamtdaten genutzt.
    
    Parameters
    ----------
    df_data : DataFrame
        Enhält alle Unfalldaten der Jahre 2016 bis 2020 inklusive der 
        Lufttemperaturwerte und der Niederschlagswerte.
    typ : str
        Ob es die Trainings/Testdaten für die Vorhersage der Unfallanzahl(num)
        oder der Unfallkategorie(cat) sein soll.
    column : str
        Spalte die Vorhergesagt werden soll.
        
    Returns
    -------
    x_train : Array
        Input des Knns.
        Enthält 80% der Daten des DataFrames df_data zum trainieren des KNNs.
    y_train : Array
        Gewünschter Output des KNNs.
        Enthält 80% der Daten des DataFrames df_data zum trainieren des KNNs.
    x_test : Array
        Daten zum Testen des Knns.
        Enthält 20% der Daten des DataFrames df_data.
    y_test : Array
        Daten zum Testen des Knns.
        Enthält 20% der Daten des DataFrames df_data..

    """
    
    y = df_data.values[:,4:5]
    x = df_data.drop(columns = [column])
    #x, y = df_data.values[:, 0:5],df_data.values[:,4]

    #Aufteilen der Daten in 80% Trainingsdaten und 20% Testdaten 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)
        
    #y_train = np.reshape(y_train, (len(y_train),1))
    
    dateiname = '../Daten/Output/x_train_' + str(stadt) + '_'  + str(typ) + '.csv'
    np.savetxt(dateiname,x_train,delimiter=",")
    
    dateiname = '../Daten/Output/y_train_' + str(stadt) + '_' + str(typ) + '.csv'
    np.savetxt(dateiname,y_train,delimiter=",")
    
    dateiname = '../Daten/Output/x_test_' + str(stadt) + '_' +  str(typ) + '.csv'
    np.savetxt(dateiname,x_test,delimiter=",")
    
    dateiname = '../Daten/Output/y_test_' + str(stadt) + '_' + str(typ) + '.csv'
    np.savetxt(dateiname,y_test,delimiter=",")

    return x_train, y_train, x_test, y_test


# Einlesen aller Daten: UNfaelle in Deutschland, Temperatur, Niederschlag und AGS-Nummern
print('Geben Sie eine Stadt/Gemeinde ein')
stadt = input() 

if stadt not in ['Mainz','Wiesbaden']:
    print("Für diese Stadt liegen leider keine Wetterdaten vor.")
    sys.exit()
    
years = [2016,2017,2018,2019,2020]
ags_gemeinden, unfaelle_de, data_niederschlag, data_temp = einlesenDaten(years,stadt)



# Reduzierung des DataFrames unfaelle_de auf eine Stadt/Gemeinde für jedes Jahr
unfaelle_region = []

for i,j in zip(unfaelle_de, years):
    unfaelle_region_j = unfaelle_re(ags_gemeinden, i, stadt, j) 
    unfaelle_region.append(unfaelle_region_j)
    
# Reduzierung von unfaelle_region auf die benötigten Daten zum Training der KNNs
u_matrix =[]
u_matrix_cat = []

for i,j in zip(unfaelle_region, years):
    u_matrix_j, u_matrix_cat_j = prepare_data(i, j)
    u_matrix.append(u_matrix_j)
    u_matrix_cat.append(u_matrix_cat_j)
    
# berechnen der durchschnittlichen Werte der Temperaturdaten und Niederschlagsdaten für 2016 bis 2021
mean_vals_temp = average_data(data_temp, data_key="TT_TU", years=years, typ='temp')
mean_vals_nied = average_data(data_niederschlag, data_key = "  R1", years = years, typ='nied')

# Zusammenführung der Daten zum Trainieren und Testen des KNN (gesamter Dataframe ohne Aufteilung in Input, Outpur, training und test)
# Unfalldaten plus Wetterdaten
data_knn_cat,data_knn_num  = merge_data(u_matrix, mean_vals_nied, mean_vals_temp, stadt, u_matrix_cat)

# Aufteilen von data_knn_cat und _num in die jeweiligen x und y Daten zum Training und Testen des KNN (zum Einlesen bereit)
x_train_cat, y_train_cat, x_test_cat, y_test_cat = data_knn(data_knn_cat, 'cat', 'category',stadt)
x_train_num, y_train_num, x_test_num, y_test_num = data_knn(data_knn_num, 'num', '# Unfaelle',stadt)

# Daten zur Vorhersage für das Jahr 2021
x_pred = pred_data()