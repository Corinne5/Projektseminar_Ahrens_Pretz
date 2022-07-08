# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:38:19 2022

@author: pretz
"""
import pandas as pd
import matplotlib.pyplot as plt

def f1_data(stadt, wetter):

    path_f1 = "../Daten/Output/f1_" + str(stadt) + "_" + str(wetter) + ".csv"  # + '_neu' + '.csv'
    f1_scores = pd.read_csv(path_f1, header=None, index_col=0)

    path_f1_label = "../Daten/Output/f1_label_" + str(stadt) + "_" + str(wetter) + ".csv"
    f1_label = pd.read_csv(path_f1_label, index_col=0)

    return f1_scores, f1_label


def f1_plot(data, title, xlabel="", ylabel=""):
    kn = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.figure(dpi=600)
    plt.plot(kn, data[1], ".-")  # ,linestyle="",marker = 'o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(kn)

def bar_plot(data, title, xlabel="", ylabel=""):
    klassen = list(range(0, len(data)))
    plt.figure(dpi=600)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.bar(klassen, data.iloc[:, 0].values)


##############Data############################
f1_scores_M_J, f1_label_M_J = f1_data("Mainz", "Ja")
f1_scores_M_N, f1_label_M_N = f1_data("Mainz", "Nein")
f1_scores_W_J, f1_label_W_J = f1_data("Wiesbaden", "Ja")
f1_scores_W_N, f1_label_W_N = f1_data("Wiesbaden", "Nein")


############### f1 in Abhängigkeit der Neuronenanzahl#########################
f1_scores_M_J = f1_scores_M_J[0:9]
f1_scores_M_N = f1_scores_M_N[0:9]
f1_scores_W_J = f1_scores_W_J[0:9]
f1_scores_W_N = f1_scores_W_N[0:9]

f1_plot(f1_scores_M_J, title="Mainz mit Wetterdaten",xlabel="Klassen Label", ylabel="f1 - Score")
f1_plot(f1_scores_M_N, title="Mainz ohne Wetterdaten",xlabel="Klassen Label", ylabel="f1 - Score")
f1_plot(f1_scores_W_J, title="Wiesbaden mit Wetterdaten",xlabel="Klassen Label", ylabel="f1 - Score")
f1_plot(f1_scores_W_N, title="Wiesbaden ohne Wetterdaten",xlabel="Klassen Label", ylabel="f1 - Score")



################ Histogram für

bar_plot(f1_label_M_J, title="Mainz mit Wetterdaten",xlabel="Klassen Label", ylabel="f1 - Score")
bar_plot(f1_label_M_N, title="Mainz ohne Wetterdaten",xlabel="Klassen Label", ylabel="f1 - Score")
bar_plot(f1_label_W_J, title="Wiesbaden mit Wetterdaten",xlabel="Klassen Label", ylabel="f1 - Score")
bar_plot(f1_label_W_N, title="Wiesbaden ohne Wetterdaten",xlabel="Klassen Label", ylabel="f1 - Score")








