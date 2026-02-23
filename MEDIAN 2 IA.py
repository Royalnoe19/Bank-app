import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. CHARGEMENT DES DONNÉES
# Note : Ton fichier utilise le point-virgule ';' comme séparateur
df = pd.read_csv('C:\\Users\\PAGE\\Documents\\DOCUMENTS ROYAL\\Banking dataset.csv', sep=';')
print(df.head())

## Suppression des donnees aberantes 









# Graphique 1 : Répartition de la cible (Y)
plt.subplot(2, 2, 1)
sns.countplot(x='y', data=df, palette='viridis')
plt.title('Répartition des Souscriptions (Cible)')

# Graphique 2 : Age vs Balance coloré par succès
plt.subplot(2, 2, 2)
sns.scatterplot(x='age', y='balance', hue='y', data=df, alpha=0.5)
plt.title('Relation Âge / Solde / Succès')

# Graphique 3 : Taux de succès par métier
plt.subplot(2, 2, 3)
sns.countplot(y='job', hue='y', data=df)
plt.title('Souscriptions par Type de Métier')

# Graphique 4 : Boxplot de la durée (Variable très importante)
plt.subplot(2, 2, 4)
sns.boxplot(x='y', y='duration', data=df)
plt.title('Impact de la Durée de l\'Appel')

plt.tight_layout()
plt.show()


# Configuration du style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. SECTEURS (Cible Y)
plt.figure()
df['y'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
plt.title('Proportion des Souscriptions (Cible Y)', fontsize=14, fontweight='bold')
plt.ylabel('') # Supprime le label 'y' sur le côté
plt.show()

# 2. HISTOGRAMME (Âge)
plt.figure()
sns.histplot(df['age'], bins=20, kde=True, color='royalblue')
plt.title('Distribution des âges des clients', fontsize=14)
plt.xlabel('Âge')
plt.ylabel('Fréquence')
plt.show()

# 3. BARRES GROUPÉES (Métier vs Souscription)
plt.figure()
sns.countplot(data=df, x='job', hue='y', palette='viridis')
plt.xticks(rotation=45)
plt.title('Souscriptions par catégorie socio-professionnelle', fontsize=14)
plt.legend(title='Souscription', labels=['Non', 'Oui'])
plt.tight_layout()
plt.show()

# 1. HEATMAP DE CORRELATION
plt.figure(figsize=(10, 8))
# On ne garde que les colonnes numériques pour la corrélation
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice de Corrélation des variables numériques')
plt.show()

# 2. VIOLIN PLOT (Education vs Balance)
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='education', y='balance', hue='y', split=True, inner="quart")
plt.ylim(-2000, 10000) # On limite pour mieux voir la densité
plt.title('Distribution du solde par niveau d\'éducation et succès')
plt.show()

# 3. FACET GRID (Age par type de logement et prêt)
g = sns.FacetGrid(df, col="housing", row="loan", margin_titles=True)
g.map(sns.histplot, "age", color="steelblue", bins=20)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Analyse croisée : Âge / Prêt Immobilier / Prêt Personnel')

plt.show()
