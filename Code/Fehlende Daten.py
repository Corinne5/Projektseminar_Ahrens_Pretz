### Fehlende Daten ###

# Imports
import pandas as pd
import seaborn as sns

# Daten einlesen
df1 = pd.read_csv("../Daten/Input/Unfallorte2016.csv", sep = ";")
df2 = pd.read_csv("../Daten/Input/Unfallorte2017.csv", sep = ";")
df3 = pd.read_csv("../Daten/Input/Unfallorte2018.csv", sep = ";")
df4 = pd.read_csv("../Daten/Input/Unfallorte2019.csv", sep = ";")
df5 = pd.read_csv("../Daten/Input/Unfallorte2020.csv", sep = ";")

# DataFrames verbinden
df = pd.concat([df1, df2, df3, df4, df5])

# Heatmap erstellen
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# Ploteinstellungen
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")