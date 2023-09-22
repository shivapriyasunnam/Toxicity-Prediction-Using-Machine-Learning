#Import all the neccesary scientific libraries
import numpy as np
import pandas as pd

#Import bioinformatics libraries
from rdkit import Chem
from rdkit.Chem import Descriptors, rdChemReactions, rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import deepchem as dc


#Importing machine learning library
import sklearn.metrics as metrics
from scipy.stats import uniform
from scipy.stats import randint 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
import os
import warnings

#import pickle library
import pickle

#This line does not print the warnings in the code
warnings.simplefilter('ignore')
currentFolder = os.getcwd() #Gets current folder path to load files from
trainFilePath = os.path.join(currentFolder, 'train_II.csv') #Training dataset file
testFilePath = os.path.join(currentFolder, 'test_II.csv') #Testing dataset file

trainingDatasetFile = 'finalTrainingFile.csv' #Name of training dataset file with features
testingDatasetFile = 'finalTestingFile.csv'#Name of testing dataset file with features

#This function retrieves the training dataset
def getTrainingData():
  if os.path.exists(trainingDatasetFile):
    X = pd.read_csv(trainingDatasetFile)
    return X, True
  else:
    X = pd.read_csv(trainFilePath)
    return X, False

#This function retrieves the molecule from given Smiles
def getMolecule(smile):
    s = Chem.MolFromSmiles(smile)
    if(s is None):
      return Chem.MolFromSmiles('')
    return s

#This function generates MACCS keys from the given SMILE
def generateMACCSKeys(smiles):
    s = Chem.MolFromSmiles(smiles)
    if(s is None):
      s = Chem.MolFromSmiles('')
    maccs = MACCSkeys.GenMACCSKeys(s)
    return maccs

#This function adds features to the given dataset, it takes two modes either train or test
def addFeatures(X, mode):
  print('\nComputing Features')
  if(mode == 'train'): #Names of the files if the mode is Training
    key = 'Id'
    molecularDescriptorsFilePath = 'MolecularDescriptorsTrain.csv'
    maccsKeysFilePath = 'MACCSKeysTrain.csv'
    circularFpFilePath = 'CircularFpTrain.csv'
    elementPropertiesFilePath = 'ElementPropertiesTrain.csv'
  if(mode == 'test'): #Names of the files if the mode is Testing
    key = 'x'
    molecularDescriptorsFilePath = 'MolecularDescriptorsTest.csv'
    maccsKeysFilePath = 'MACCSKeysTest.csv'
    circularFpFilePath = 'CircularFpTest.csv'
    elementPropertiesFilePath = 'ElementPropertiesTest.csv'

  #Retrieving the SMILES and Assays seperately
  X['Smile'] = X[key].apply(lambda x: x.split(';', 1)[0])
  X['Assay'] = X[key].apply(lambda x: int(x.split(';', 1)[1]))
  
  #Initializing empty dataframes for generating features
  moldesc = pd.DataFrame()
  maccskeys = pd.DataFrame()
  circularfp = pd.DataFrame()

  #Generating the Molecular Descriptors
  molecularDescriptorsFile = os.path.join(currentFolder, molecularDescriptorsFilePath)
  if os.path.exists(molecularDescriptorsFile):
    moldesc = pd.read_csv(molecularDescriptorsFilePath)
    print('\nMolecular Descriptors already computed for ', mode, 'Dataset')
  else:
    print('\nComputing Molecular Descriptors....')
    descriptors = []
    for smile in X['Smile']:
      mol = getMolecule(smile)
      try:
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x 
                                                                  in Descriptors._descList])
        molDescriptors = calc.CalcDescriptors(mol)
      except:
        print('Error Occured while calculating Molecular Descriptors')
      descriptors.append(molDescriptors)
    molDescDF = pd.DataFrame(descriptors, columns=calc.GetDescriptorNames())
    molDescDF.to_csv(molecularDescriptorsFilePath, index = False)
    moldesc = molDescDF 
    print('Computations complete!')
  print('Number of Molecular Descriptors Computed : ', moldesc.columns.size)

  # Generating the MACCS keys
  maccsKeysFile = os.path.join(currentFolder, maccsKeysFilePath)
  if os.path.exists(maccsKeysFile):
    maccskeys = pd.read_csv(maccsKeysFilePath)
    print('\nMACCS Keys already computed for ', mode, 'Dataset')
  else:
    print('\nComputing MACCS Features...............')
    mDF = pd.DataFrame()
    mDF['MACCSKeys'] = X['Smile'].apply(generateMACCSKeys)
    keyLabels = [list(mDF['MACCSKeys'][i]) for i in range(len(X))]
    finalMDF = pd.DataFrame(keyLabels, columns=[f'maccs_{i}' for i in range(167)])
    finalMDF.to_csv(maccsKeysFilePath, index = False)
    maccskeys = finalMDF
    print('Computations complete!')
  print('MACCS Fingerprint Bits : ', maccskeys.columns.size)

  # Generating the Deepchem keys
  circularFpFile =  os.path.join(currentFolder, circularFpFilePath)
  if os.path.exists(circularFpFile):
    circularfp = pd.read_csv(circularFpFilePath)
    print('\nCircular Fingerprint already computed for ', mode, 'Dataset')
  else:
    print('\nComputing Circular Fingerprint Features for Dataset....')
    featurizer = dc.feat.CircularFingerprint()
    features = featurizer.featurize(X['Smile'])
    circularfp = pd.DataFrame(np.array(features))
    circularfp.columns = ['feature_' + str(i) for i in range(circularfp.shape[1])]
    circularfp.to_csv(circularFpFilePath, index = False)
    print('Computations complete!')
  print('Circular Fingerprint Bits : ', circularfp.columns.size )

  #Concatenating all the features into one dataframe
  X = pd.concat([X, moldesc], axis=1)
  X = pd.concat([X, maccskeys], axis=1)
  X = pd.concat([X, circularfp], axis=1)
  print('Number of Features (inclusing Fingerprint Bits) are : ', X.columns.size)
  print('\nSaving Training Dataset')
  #Saving the datasets generated
  if(mode == 'train'):
   X.to_csv('FinalTrainingDataset.csv', index = False)
   print('Saved Final Training Dataset File')
  if(mode == 'test'):
   X.to_csv('FinalTestingDataset.csv', index = False)
   print('Saved Final Testing Dataset File')
  return X

#This function gets the Testing Dataset file
def getTestingData():
  if os.path.exists(testingDatasetFile):
    print('\nTesting Dataset with Features already Computed')
    X = pd.read_csv(testingDatasetFile)
    return X, True
  else:
    X = pd.read_csv(testFilePath)
    print('\nMust Calculate Features for Testing Data......')
    return X, False

#This function should be called to get Training Dataset
def getTrainingDataset():
  X, featuresAdded = getTrainingData()
  print('\nFeatures Computed : ', featuresAdded)
  X['Expected'].replace({1: 0, 2: 1}, inplace=True)
  if(featuresAdded is False):
    X = addFeatures(X, 'train')
  y = X['Expected']
  X = X.drop(['Expected', 'Smile', 'Id'], axis=1)
  X = X.fillna(0)
  print('Training Dataset Ready!')
  return X, y


#Call this function to get Predictions on Test File for Submission. 
def getSubmissionPrediction(model, name):
  print('\nComputing Results for Kaggle Submission')
  XOutputTest, featuresAdded = getTestingData()
  if(featuresAdded is False):
    XOutputTest = addFeatures(XOutputTest, 'test')
  x = XOutputTest['x']
  XOutputTest = XOutputTest.drop(['Smile', 'x'], axis=1)
  XOutputTest = XOutputTest.fillna(0)
  predictions = predictOutput(model, XOutputTest)
  predictions_mod = [1 if x == 0 else 2 if x == 1 else x for x in predictions]
  output = pd.DataFrame({'Id': x, 'Predicted': predictions_mod})
  output.to_csv('kaggle' + model.__class__.__name__ + name + '.csv', index=False)
  print('Results Computed and Saved to file ','kaggle' + model.__class__.__name__ + name + '.csv' )
  return output

#Predicts outputs for the given model and returns it as a dataframe
def predictOutput(model, XTest):
  predictions = model.predict(XTest) #Predicts data from model for XTest values
  return predictions

#Machine Learning Functions
def splitDataset(X, y, size):
  return train_test_split(X, y, test_size=size)

def splitDatasetKFold(X, y, k):
  kf = KFold(n_splits=k, shuffle=True, random_state=42)
  return kf.split(X, y)

#Calculates the metrics for the model
def calculateMetrics(y_test, y_pred, model):
  print('\nF1 Score         : %.2f' % metrics.f1_score(y_test, y_pred)) #Calculates F1 Score
  print('Accuracy Score   : %.2f' % metrics.accuracy_score(y_test,y_pred)) #Calculates Accuracy
  print('Precision Score  : %.2f' % metrics.precision_score(y_test, y_pred)) #Calculates Precision
  print('Recall Score     : %.2f' % metrics.recall_score(y_test, y_pred)) #Calculates Recall
  print('ROC AUC Score    : %.2f' % metrics.roc_auc_score(y_test, y_pred)) #Calculates ROC AUC Score


#Run the Machine Learning Models
def runClassifierModel(X, y, datsetSplitValue, model):
  print("\n"+ model.__class__.__name__ + "\n") #Print Model Name
  X_train, X_test, y_train, y_test = splitDataset(X, y, datsetSplitValue) #Split the dataset into Training and Testing 
  model.fit(X_train, y_train) #Fit the Model with Training Data
  y_pred = model.predict(X_test) #Predict the ouputs for Test Data
  calculateMetrics(y_test, y_pred, model) #Calculate metrics for the model
  return model

#This function is to perform K fold cross validation
def runClassifierModelKFold(X, y, k, model, name):
  print('\nStarting Classifier Training with KFold Cross Validation')
  f1 = [] 
  i = 1
  kf = KFold(n_splits=k, shuffle=True, random_state=42)
  for train_index, test_index in kf.split(X, y):
    print('\nRunning Fold Number :', i)
    i = i+1
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    getSubmissionPrediction(model, name + 'Kfold' + str(i-1))
    y_pred = model.predict(X_test)
    calculateMetrics(y_test, y_pred, model)
    f1.append(metrics.f1_score(y_test, y_pred))
  print('Model Training Completed Successfully')
  return model, f1

#This function performs Randomized Search to get best parameters
def doRandomizedSearch(X, y, model, parameters):
  print('\nStarting Hyperparameter Tuning using Randomized Search CV')
  rsModel = RandomizedSearchCV(estimator=model, verbose = 2, param_distributions=parameters, n_iter=2, cv=5, random_state=42) #Call the Randomizer with the parameters
  rsModel.fit(X, y) #Fits the model
  print("Best Parameters for ", model.__class__.__name__ +" are \n", rsModel.best_params_) #Prints the best Parameters
  return rsModel.best_params_ #Returns the best parameters
  
#Resampling the Dataset
def randomOverSampling(X, y):
  print('\nDataset Before Balancing')
  print("Number of 0s  : ", (y == 0).sum())
  print("Number of 1s  : ", (y == 1).sum())
  print("Total Samples : ", (y == 0).sum() + (y == 1).sum() )
  oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
  X, y = oversampler.fit_resample(X, y)
  print('After Balancing Data Using RANDOM OVER SAMPLER')
  print("Number of 0s  : ", (y == 0).sum())
  print("Number of 1s  : ", (y == 1).sum())
  print("Total Samples : ", (y == 0).sum() + (y == 1).sum() )
  return X, y
  
#Dimensionality Reduction
from sklearn.decomposition import PCA
def doPCA(X):
  pca = PCA(n_components=500)
  X = pca.fit_transform(X)
  return X

#This function uses pickle to save the model
def saveModel(model, filename):
  pickle.dump(model, open(filename, 'wb'))
  print('Saved ML Model after Training: ', filename)

#-----------------------------------------------------------------------------------------------------------------#
#Execution begins here

runRCV = False
trainModel = False
pickModel = True

#Assign the runRCV variable to True if you want to generate the best parameters

if(runRCV):
  XGBParams = {
      'max_depth': randint(3, 15),
      'learning_rate': uniform(0.01, 0.3),
      'n_estimators': randint(100, 1000),
      'gamma': uniform(0, 1),
      'subsample': uniform(0.1, 0.5),
      'colsample_bytree': uniform(0.1, 0.5),
      'min_child_weight': randint(1, 10),
      'reg_alpha': uniform(0.1, 0.5),
      'reg_lambda': uniform(0.1, 0.5)
  }
  model = XGBClassifier()
  doRandomizedSearch(X, y, model, XGBParams)

#This code snippet Trains the model from begining
if(trainModel):
  X, y = getTrainingDataset()
  X, y = randomOverSampling(X, y)
  print('\n')
  print('Running Machine Learning Models')

  print('XGB with usual parameters')
  model =  XGBClassifier(n_estimators=761, learning_rate=0.19033450352296263, max_depth=10, 
                        random_state=42, gamma=0.8661761457749352, subsample = 0.5282057895135501, 
                        colsample_bytree = 0.5290418060840998, reg_alpha = 1.0,
                        reg_lambda = 0.1, min_child_weight = 3)
  print('Starting K fold Classification ')
  model, f1 = runClassifierModelKFold(X, y, 8, model, 'ALLF')
  saveModel(model, 'XGBClassifierALLF')
  print('Final F1 Score of the Model : ', np.mean(f1))
  print('Generating submission file for Kaggle')
  output = getSubmissionPrediction(model, 'ALLF')

#This code snippet will pick Pickle model to generate predictions
if(pickModel):
  print('\nGenerating submission file for Kaggle from Pretrained Model')
  print('The model used is XGBClassifier with K-Fold Cross Validation')
  print('\nLoading Model')
  xgb = pickle.load(open( 'XGBModelPretrained', 'rb'))
  print('Model Loaded!')
  getSubmissionPrediction(xgb, 'testprediction')