import json
from pathlib import Path
import matplotlib.pyplot as plt
import os
import shutil

def histogramme(path):
    p = Path(path)  
    valeurs = {}
    for label in p.iterdir(): 
        with open(str(label), "r") as jsonfile: 
            data = json.load(jsonfile)
            for shape in data['shapes']:
                if shape['label'] in valeurs:
                    valeurs[shape['label']]+=1
                else :
                    valeurs[shape['label']]=1
    
    plt.bar(range(len(valeurs)), list(valeurs.values()), align='center')
    plt.xticks(range(len(valeurs)), list(valeurs.keys()))

#histogramme('../bases/base_test/labels_test')
histogramme('../bases/base_validation/labels_validation')