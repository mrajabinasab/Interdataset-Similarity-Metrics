import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def file_exists(filepath):
    return os.path.isfile(filepath)

def dataset_loader(dataset):
    if file_exists("./datasets/" + dataset + "_X.csv") and file_exists("./datasets/" + dataset + "_y.csv"):
        X = pd.read_csv("./datasets/" + dataset + "_X.csv")
        y = pd.read_csv("./datasets/" + dataset + "_y.csv")
    else:
        if dataset == "ionosphere":
            ionosphere = fetch_ucirepo(id=52)  
            X = ionosphere.data.features 
            y = ionosphere.data.targets
        elif dataset == "glass":
            glass_identification = fetch_ucirepo(id=42) 
            X = glass_identification.data.features 
            y = glass_identification.data.targets 
        elif dataset == "iris":
            iris = fetch_ucirepo(id=53) 
            X = iris.data.features 
            y = iris.data.targets 
        elif dataset == "yeast":
            yeast = fetch_ucirepo(id=110) 
            X = yeast.data.features 
            y = yeast.data.targets 
        elif dataset == "image":
            image_segmentation = fetch_ucirepo(id=50) 
            X = image_segmentation.data.features 
            y = image_segmentation.data.targets
        elif dataset == "hepatitis":
            hepatitis = fetch_ucirepo(id=46) 
            X = hepatitis.data.features 
            y = hepatitis.data.targets
        elif dataset == "cylinder":
            cylinder_bands = fetch_ucirepo(id=32) 
            X = cylinder_bands.data.features 
            y = cylinder_bands.data.targets
        elif dataset == "audiology":
            audiology_standardized = fetch_ucirepo(id=8) 
            X = audiology_standardized.data.features 
            y = audiology_standardized.data.targets 
        elif dataset == "credit":
            credit_approval = fetch_ucirepo(id=27) 
            X = credit_approval.data.features 
            y = credit_approval.data.targets
        elif dataset == "dermatology":
            dermatology = fetch_ucirepo(id=33) 
            X = dermatology.data.features 
            y = dermatology.data.targets  
        elif dataset == "automobile":
            automobile = fetch_ucirepo(id=10) 
            X = automobile.data.features 
            y = automobile.data.targets 
        elif dataset == "breast_cancer":
            breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
            X = breast_cancer_wisconsin_diagnostic.data.features 
            y = breast_cancer_wisconsin_diagnostic.data.targets   
        elif dataset == "lymphography":
            lymphography = fetch_ucirepo(id=63) 
            X = lymphography.data.features 
            y = lymphography.data.targets 
        elif dataset == "zoo":
            zoo = fetch_ucirepo(id=111) 
            X = zoo.data.features 
            y = zoo.data.targets
        elif dataset == "wine":
            wine = fetch_ucirepo(id=109) 
            X = wine.data.features 
            y = wine.data.targets 
        elif dataset == "tictactoe":
            tic_tac_toe_endgame = fetch_ucirepo(id=101) 
            X = tic_tac_toe_endgame.data.features 
            y = tic_tac_toe_endgame.data.targets 
        elif dataset == "primary_tumor":
            primary_tumor = fetch_ucirepo(id=83) 
            X = primary_tumor.data.features 
            y = primary_tumor.data.targets
        elif dataset == "horse_colic":
            horse_colic = fetch_ucirepo(id=47)  
            X = horse_colic.data.features 
            y = horse_colic.data.targets
        elif dataset == "lung_cancer":
            lung_cancer = fetch_ucirepo(id=62) 
            X = lung_cancer.data.features 
            y = lung_cancer.data.targets
        elif dataset == "raisin":
            raisin = fetch_ucirepo(id=850) 
            X = raisin.data.features 
            y = raisin.data.targets
        elif dataset == "diabetic_retino":
            diab = fetch_ucirepo(id=329) 
            X = diab.data.features 
            y = diab.data.targets
        elif dataset == "phishing":
            phishing = fetch_ucirepo(id=379) 
            X = phishing.data.features 
            y = phishing.data.targets     
        elif dataset == "steel":
            steel = fetch_ucirepo(id=198) 
            X = steel.data.features 
            y = steel.data.targets     
        elif dataset == "nursery":
            nursery = fetch_ucirepo(id=76) 
            X = nursery.data.features 
            y = nursery.data.targets     
        elif dataset == "rice":
            rice = fetch_ucirepo(id=545) 
            X = rice.data.features 
            y = rice.data.targets     
        elif dataset == "drybean":
            drybean = fetch_ucirepo(id=602) 
            X = drybean.data.features 
            y = drybean.data.targets     
        elif dataset == "car_eval":
            car = fetch_ucirepo(id=19) 
            X = car.data.features 
            y = car.data.targets     
        elif dataset == "letter":
            letter = fetch_ucirepo(id=59) 
            X = letter.data.features 
            y = letter.data.targets             
        else:
            print("Requested dataset does not exist.")
        X.to_csv("./datasets/" + dataset + "_X.csv", index=False)
        y.to_csv("./datasets/" + dataset + "_y.csv", index=False)
    return X, y