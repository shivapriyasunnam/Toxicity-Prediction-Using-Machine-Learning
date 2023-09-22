# Toxicity Prediction of Chemicals using Data Mining and Machine Learning

**1. Introduction**

This project focuses on building a Machine Learning solution for predicting the toxicity of a chemical when
the respective SMILE molecule is provided. I have experimented with many machine learning models and
techniques to achieve an accurate model. This report talks about the steps followed in the project during
the development of the solution.

**2. Data Preparation**

We were provided with a Training and a Testing Dataset, these datasets were worked on and used to train
and test the machine learning algorithms used for Toxicity Prediction.


***2.1 Training Dataset***

The training dataset consisted of 75,384 samples. Each sample had an ‘ID’ and an ‘Expected’ value. The ID
was a combination of SMILE molecule of the compound and the Assay ID, separated with a semicolon.
SMILE is a Simplified Molecular Input Line; this format is used for computer programming applications.
The Assay ID is a unique identifier that represents scientific tests. The first step towards preparing training
data was to separate the SMILE molecules from the Assay IDs. After being separated the Assay ID was
taken as a feature and given to the training model.

***2.2 Testing Dataset***
The testing dataset consisted of 10,995 samples. They were in a similar format to the training dataset.
However, the result ‘Expected’ was not present as this dataset should be used for testing purposes only.
The samples were provided with ‘x’ as the column identifier. These samples were also separated using the
semicolon before giving them to the model. Assay ID was also considered as a feature similar to the testing
dataset.

**3. Feature Identification**

Using the SMILE molecule information, one can retrieve much information from the web. There are many
libraries available in Python and other languages that provide such information with simple commands.


***3.1 Libraries in Python for Chemical Features***

Libraries such as RDKit, DeepChem, PubChem, MolIVS, ChemSpider and Pybel are few examples for getting
chemical information using just a SMILE molecule. They are some great sources with clear documentation
about every function. In this project, I specifically used the RDKit and PubChem libraries to finalize my
features.


***3.2 Final Training Features***

I have used many combinations of features and also tried to do feature selection using algorithms like
Recursive Feature Elimination. Finally, the features that contributed to my best score were Molecular
Descriptors from RDKit, MACCS Fingerprint (166 Bit) from RDKIT and Element Property Fingerprint (
Bit) from DeepChem.

The molecular descriptors give much chemical information in numerical form, some examples of
descriptors are Number of Valence Electrons, Number of Radical Electrons and Heavy Atom Weight. During
the retrieval of molecular descriptors, some values from the dataset threw some errors, they were


discarded temporarily. The MACCS keys are Molecular Access System fingerprints that provide information
about certain structural properties in the molecules. Many applications use these fingerprints for
clustering and classification tasks. MACCS keys have contributed to increasing the total accuracy of the
predictions. The Element Structure Property keys were also retrieved from DeepChem library. These
fingerprints provide information about physicochemical properties. Properties such as polarizability and
electronegativity are also given in these keys.

**4. Dataset Balancing – Random Over Sampler**

After extracting all the features and training the machine learning models, the accuracy was still less and
not improving. This clearly meant that the dataset was skewed and had to be balanced before feeding it
to the model. Another change I made to the dataset was to convert the output numerical into
standardized format. The original output values were either ‘1’ or ‘2’ but I transformed that to either ‘0’
or ‘1’. I also then balanced the originally skewed dataset using **_RandomOverSampler_** this is a balancing
model available on sklearn.


I used a sampling strategy as 0.5 and random state as 42. The sampling strategy is the ratio of minority
class to majority class. I tried with multiple float values such as 0.2, 0.3, 0.4, 0.6 and 0.7 but 0.5 worked
best for my experiment. After random over sampling, the total number of samples that can be fed to
training model was now 97,078. These samples were not imbalanced or skewed. Our training dataset is
now ready to be fed to any machine learning algorithm!

**5. Model Selection**

Examining the problem more closely, it is clear that this is a Classification problem because the datasets
provided are in such a way that when a model is given input data, here a chemical molecule, the model
has to predict if that molecule is toxic or not. The output classes here are ‘1’ for toxic prediction and ‘0’
for non-toxic or safe prediction.

***5.1 XGB Classifier***

I experimented with many classification models and tried the following experiments on all the models.
However, in this entire document, I will be talking about only one model that worked best for me and with
which I achieved my best score. The model that I scored best for me is XGB or the e **_Xtreme Gradient
Boosting Classifier Model_**. The main reason I think XGB worked best for me is that unlike most models,
XGB is an ensemble model, it internally combines many simple models. XGB also has the ability to handle
missing data internally. Additionally, it also knows how to work with relationships that are not linear and
computes regularization for the dataset. XGB Classifier also is extremely scalable. After balancing my
dataset the size of the dataset was almost 1 lakh samples with hundreds of features. XGB can handle such
a huge dataset with ease. Some other models that I performed experiments with are Random Forests,


Naives Bayes, Support Vector Machines, Decision Trees and Gaussian Naives Bayes. However, all of them
performed either slightly similar in comparison to XGB or worse. I also tried to create ensemble models
that were trained on different feature sets but did not perform as good as XGB.

**6. Training**

Training the model is the most important phase in developing a machine learning solution. At first, I used
some default parameters to train the model that resulted in low accuracies which ranged between 60 - 70.
Researching more about the parameters, it was important that the parameters had to be set to
appropriate values based on your training dataset for best results.

***6.1 Hyperparameter Tuning***

After balancing the data, the same dataset was passed to a Hyperparameter tuning model called the
‘Randomized Search CV’. Randomized Search CV is a technique used in machine learning to find the best
hyperparameters for a given model. It is a type of cross-validation that randomly selects hyperparameters
from a specified range of values and evaluates the model's performance using those values. This method
is particularly useful when the hyperparameter space is large and exhaustive search is impractical.


The following are the parameters that are accepted by XGB Classifier:

- max_depth: The maximum possible depth of the Tree.
- learning_rate: The rate at which the algorithm learns.
- n_estimators: Number of estimators for the model.
- gamma: The value for loss reduction in the model.
- subsample: Ratio of observations that are sampled randomly.
- colsample_bytree: Ratios of coloumns that are sampled randomly.
- min_child_weight: Min weight for each child node.


- reg_alpha: Regularization value for weights.
- reg_lambda: Regularization value for weights L2.
    I have run randomized search CV for 2 iterations with 5 cross validation folds. Once, the
    parameters are generated, use them in XGB Classifier to train your model. I have used cross
    validation training for 8 folds to achieve my best score on Kaggle. Will explain more about cross
    validation in the next section. Finally, the score I got after following all the above steps are as
    follows.

**7. Testing**

I used the k fold Cross validation techniques to perform testing. The idea behind k-fold cross-validation is
to split the original dataset into k subsets (or "folds"), then train and test the model k times, using a
different fold as the testing set in each iteration and the remaining folds as the training set. I have used 8
folds to get my result.


***7.1 Internal Validation***

I have used 5 metrics for internal validation; however, the F1 Score was the major determinant of
understanding the performance of my model. The following are the score I referred to:

- F1 Score
- Accuracy Score
- Precision Score
- Recall Score
- ROC AUC Score
  
**8. Saving the Model using Pickle**

Once, the training and testing functions complete execution, I have saved the trained model using Pickle
for predicting future results simulating the model again and again to go through the processing of data
preparation and training. The code submit picks up the pickle model and generates the submission file
until states otherwise in the code. To run the entire code we have to change the value of _trainModel_
variable to True.

**9. Kaggle Leaderboard**

The first BEST Private Score is the best score with 83.02 

**10. Reproducibility**

Docker is a containerization technology for building, testing, and deploying applications quickly. To run the
the project, one can simply run the **_Dockerfile_** created for the project that’s available in the GitHub Repository.

**Operating System** : Windows

**Python** : Version 3 and above

**Libraries Required** :

pandas

rdkit

scikit-learn

xgboost

imblearn

deepchem

tensorflow

**Running the Code with DOCKER:**

Simply run the 'run.sh' file that executes Docker Commands
    - Dockerfile automatically installs the required libraries

**Docker Commands:**

docker build -t toxicity -f Dockerfile .

docker images

docker run toxicity

**Run in Local:**

Install the above libraries using
    pip install {library}
Run the python command
    python toxicityPrediction.py


**11. Conclusion**

Through this project I was able to implement a full-fledged Machine Learning Solution for predicting the
toxicity of a Molecule when a SMILE compound is provided. I scored 83.02 private accuracy and 80.
public accuracy. I also reproduced the same code using Dockerfile.


