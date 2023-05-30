# hides all warnings
import warnings
warnings.filterwarnings('ignore')
# imports
# pandas 
import pandas as pd
# numpy
import numpy as np
# matplotlib 
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 5)
# sns
import seaborn as sns
# util
import utils


##############################################################
# Read Data 
##############################################################

# read dataset
print("\n*** Read Data ***")
df = pd.read_csv('./data/farmingham.csv')
# https://docs.python.org/3/library/codecs.html#standard-encodings
# df = pd.read_csv('./data/data-file.csv', encoding = "ISO-8859-1")
print("Done ...")


##############################################################
# Exploratory Data Analysis
##############################################################

# rows & cols
print("\n*** Rows & Cols ***")
print("Rows",df.shape[0])
print("Cols",df.shape[1])

# columns
print("\n*** Column Names ***")
print(df.columns)

# data types
print("\n*** Data Types ***")
print(df.dtypes)

# count of unique values
print("\n*** Unique Values ***")
print(df.apply(lambda x: x.nunique()))

# summary numeric cols
print("\n*** Summary Numeric ***")
dsTypes = df.dtypes
if "int" in dsTypes.tolist() or "float" in dsTypes.tolist():
    print(df.describe(include=np.number))
else:
    print("None ...")

# summary object cols
print("\n*** Summary AlphaNumeric ***")
dsTypes = df.dtypes
if "object" in dsTypes.tolist():
    print(df.describe(include=object))
else:
    print("None ...")

# head
print("\n*** Head ***")
print(df.head())

# info
print("\n*** Structure ***")
print(df.info())


##############################################################
# Class Variable & Counts
##############################################################

# store class variable  
# change as required
clsVars = "TenYearCHD"
print("\n*** Class Vars ***")
print(clsVars)

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# # class variable - convert string / categoric to numeric
# # change as required
# print("\n*** Class Vars - Alpha to Numeric ***")
# print(df['col_2'].unique())
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# df['col_2'] = le.fit_transform(df['col_2'])
# print(df['col_2'].unique())


##############################################################
# Data Transformation
##############################################################

# # drop cols
# # identifiers
# # nominals
# # descriptors
# # change as required
# print("\n*** Drop Cols ***")
# df = df.drop('col_0', axis=1)
# print("Done ...")

# # transformations - convert object to float
# # change as required
# print("\n*** Transformations ***")
# colNames = ['colName']
# for colName in colNames:
#       df[colName] = pd.to_numeric(df[colName], errors = "coerce")
# print("Done ...")

# # object variable - convert object to numeric
# # change as required
# colName = "col_2"
# print("\n*** " + colName + " - Alpha to Numeric ***")
# print(df[colName].unique())
# from sklearn import preprocessing
# leCol = preprocessing.LabelEncoder()
# df[colName] = leCol.fit_transform(df[colName])
# print(df[colName].unique())

# check outlier count
print('\n*** Outlier Count ***')
print(utils.OutlierCount(df))

# check outlier row index 
print('\n*** Outlier Index ***')
print(utils.OutlierIndex(df))

# check outlier values
print('\n*** Outlier Values ***')
print(utils.OutlierValues(df))

# handle outliers if required
# change as required
print('\n*** Handle Outliers ***')
df = utils.HandleOutliers(df, clsVars)
print("Done ...")

# check zeros
print('\n*** Columns With Zeros ***')
print((df==0).sum())

# # handle zeros if required
# # change as required
# print('\n*** Handle Zeros ***')
# dfp = utils.HandleZeros(df, ['col_3','col_7'])
# print("Done ...")

# drop col if all values are same
print("\n*** Same Value Cols Drop ***")
lDropCols = utils.SameValuesCols(df, clsVars, 1)
print(lDropCols)
if lDropCols != []:
    df = df.drop(lDropCols, axis=1)
print("Done ...")

# drop col if contains 100% unique values
print("\n*** Uniq Value Cols Drop ***")
lDropCols = utils.UniqValuesCols(df, clsVars, 1)
print(lDropCols)
if lDropCols != []:
    df = df.drop(lDropCols, axis=1)
print("Done ...")

# drop col if more than 50% null values
print("\n*** Null Value Cols Drop ***")
lDropCols = utils.NullValuesCols(df, clsVars, 0.50)
print(lDropCols)
if lDropCols != []:
    df = df.drop(lDropCols, axis=1)
print("Done ...")

# check nulls
print('\n*** Columns With Nulls ***')
print(df.isnull().sum()) 

# handle nulls if required
# change as required
print('\n*** Handle Nulls ***')
df = utils.HandleNullsWithMean(df, clsVars)
print("Done ...")

# check variance
print('\n*** Variance In Columns ***')
print(df.var())

# check std dev 
print('\n*** StdDev In Columns ***')
print(df.std())

# # handle normalization if required
# # change as required
# print('\n*** Normalize Data ***')
# df = utils.NormalizeData(df, clsVars)
# print('Done ...')

# feature selection
print("\n*** Feature Scores - XTC ***")
print(utils.getFeatureScoresXTC(df, clsVars))

# print("\n*** Feature Scores - SKB ***")
print(utils.getFeatureScoresSKB(df, clsVars))

# select feature to drop 
# # change as required
print("\n*** Features To Drop ***")
lDropCols = utils.DropFeaturesRank(df, clsVars, 1)
print(lDropCols)
if lDropCols != []:
    df = df.drop(lDropCols, axis=1)
print("Done ...")


##############################################################
# Visual Data Anlytics
##############################################################

# boxplot
print('\n*** Boxplot ***')
colNames = df.columns.tolist()
colNames.remove(clsVars)
for colName in colNames:
    plt.figure()
    sns.boxplot(y=df[colName], color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()

# histograms
# plot histograms
print("\n*** Histogram Plot ***")
colNames = df.columns.tolist()
colNames.remove(clsVars)
for colName in colNames:
    colValues = df[colName].values
    plt.figure()
    sns.distplot(colValues, bins=7, kde=False, color='b')
    plt.title(colName)
    plt.ylabel(colName)
    plt.xlabel('Bins')
    plt.show()
    
# class count plot
# change as required
vMaxCats = 10
colNames = df.columns.tolist()
colNames.remove(clsVars)
print("\n*** Distribution Plot ***")
bFlag = False
for colName in colNames:
    if len(df[colName].unique()) > vMaxCats:
        continue
    plt.figure()
    sns.countplot(df[colName],label="Count")
    plt.title(colName)
    plt.show()
    bFlag = True
if bFlag==False:
    print("No Categoric Variables Found")

# check class
# outcome groupby count    
print("\n*** Group Counts ***")
print(df.groupby(clsVars).size())
print("")

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df[clsVars],label="Count")
plt.title('Class Variable')
plt.show()


################################
# Classification 
# set X & y
###############################

# split into data & target
print("\n*** Prepare Data ***")
allCols = df.columns.tolist()
print(allCols)
allCols.remove(clsVars)
print(allCols)
X = df[allCols].values
y = df[clsVars].values

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))

# head
print("\n*** Prepare Data - Head ***")
print(X[0:4])
print(y[0:4])

# counts
print("\n*** Counts ***")
print(df.groupby(df[clsVars]).size())

# over sampling
# change as required
print("\n*** Over Sampling Process ***")
X, y = utils.getOverSamplerData(X, y)
print("Done ...")

# counts
print("\n*** Counts ***")
unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# shape
print("\n*** Prepare Data - Shape ***")
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))


################################
# Classification - init models
###############################

# original
# import all model & metrics
print("\n*** Importing Models ***")
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
from sklearn.naive_bayes import GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
print("Done ...")

# create a list of models so that we can use the models in an iterstive manner
print("\n*** Creating Models ***")
lModels = []
lModels.append(('Random Forest  ', RandomForestClassifier(random_state=707)))
lModels.append(('SVM-Classifier ',  SVC(random_state=707)))
lModels.append(('KNN-Classifier ', KNeighborsClassifier()))
lModels.append(('LogRegression  ', LogisticRegression(random_state=707)))
lModels.append(('DecisionTree   ', DecisionTreeClassifier(random_state=707)))
lModels.append(('NaiveBayes     ', GaussianNB()))
for vModel in lModels:
    print(vModel)
print("Done ...")


################################
# Classification - cross validation
###############################

# blank list to store results
print("\n*** Cross Validation Init ***")
xvModNames = []
xvAccuracy = []
xvSDScores = []
print("Done ...")

# cross validation
from sklearn import model_selection
#print("\n*** Cross Validation ***")
# iterate through the lModels
for vModelName, oModelObj in lModels:
    # select xv folds
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=707)
    # actual corss validation
    cvAccuracy = model_selection.cross_val_score(oModelObj, X, y, cv=kfold, scoring='accuracy')
    # prints result of cross val ... scores count = lfold splits
    print(vModelName,":  ",cvAccuracy)
    # update lists for future use
    xvModNames.append(vModelName)
    xvAccuracy.append(cvAccuracy.mean())
    xvSDScores.append(cvAccuracy.std())
    
# cross val summary
print("\n*** Cross Validation Summary ***")
# header
msg = "%10s: %10s %8s" % ("Model   ", "xvAccuracy", "xvStdDev")
print(msg)
# for each model
for i in range(0,len(lModels)):
    # print accuracy mean & std
    msg = "%10s: %5.7f %5.7f" % (xvModNames[i], xvAccuracy[i], xvSDScores[i])
    print(msg)

# find model with best xv accuracy & print details
print("\n*** Best XV Accuracy Model ***")
xvIndex = xvAccuracy.index(max(xvAccuracy))
print("Index      : ",xvIndex)
print("Model Name : ",xvModNames[xvIndex])
print("XVAccuracy : ",xvAccuracy[xvIndex])
print("XVStdDev   : ",xvSDScores[xvIndex])
print("Model      : ",lModels[xvIndex])


################################
# Classification 
# Split Train & Test
###############################

# imports
from sklearn.model_selection import train_test_split

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                test_size=0.2, random_state=707)

# shapes
print("\n*** Train & Test Data ***")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counts
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("\n*** Frequency of unique values of Train Data ***")
print(np.asarray((unique_elements, counts_elements)))

# counts
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("\n*** Frequency of unique values of Test Data ***")
print(np.asarray((unique_elements, counts_elements)))


################################
# Classification- Create Model
###############################

print("\n*** Accuracy & Models ***")
print("Cross Validation")
print("Accuracy:", xvAccuracy[xvIndex])
print("Model   :", lModels[xvIndex]) 

# classifier object
# select model with best acc
print("\n*** Classfier Object ***")
model = lModels[xvIndex][1]
print(model)
# fit the model
model.fit(X_train,y_train)
print("Done ...")


################################
# Classification  - Predict Test
# evaluate : Accuracy & Confusion Metrics
###############################

# classifier object
print("\n*** Predict Test ***")
# predicting the Test set results
p_test = model.predict(X_test)            # use model ... predict
print("Done ...")

# accuracy
from sklearn.metrics import accuracy_score
print("\n*** Accuracy ***")
accuracy = accuracy_score(y_test, p_test)*100
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
from sklearn.metrics import confusion_matrix
print("\n*** Confusion Matrix - Original ***")
cm = confusion_matrix(y_test, y_test)
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
print("\n*** Confusion Matrix - Predicted ***")
cm = confusion_matrix(y_test, p_test)
print(cm)

# classification report
from sklearn.metrics import classification_report
print("\n*** Classification Report ***")
cr = classification_report(y_test,p_test)
print(cr)

#------------------------------------------

#### RAW RESULTS ####

# *** Predict Test ***
# Done ...
# *** Accuracy ***
# 84.90566037735849

#------------------------------------------

#### RESULTS WITH OVERSAMPLING ####

# *** Predict Test ***
# Done ...
# *** Accuracy ***
# 97.77623349548298

#------------------------------------------

#### RESULTS WITH OVERSAMPLING AND FEATURE SELECTION ####

# *** Predict Test ***
# Done ...
# *** Accuracy ***
# 97.91521890201528

#------------------------------------------ BEST RESULTS

#### RESULTS WITH OVERSAMPLING, FEATURE SELECTION & OUTLIER ####

# *** Predict Test ***
# Done ...
# *** Accuracy ***
# 98.0542043085476

#------------------------------------------

#### RESULTS WITH OVERSAMPLING AND OUTLIER ####

# *** Predict Test ***
# Done ...
# *** Accuracy ***
# 97.77623349548298



# make dftest
# only for show
# not to be done in production
print("\n*** Recreate Test ***")
dfTest =  pd.DataFrame(data = X_test)
dfTest.columns = allCols
dfTest[clsVars] = y_test
dfTest['Predict'] = p_test
# dfTest[clsVars] = le.inverse_transform(dfTest[clsVars])
# dfTest['Predict'] = le.inverse_transform(dfTest['Predict'])
print("Done ...")



###################################
#
#Saving the model
#
###################################

# classifier object
# select the best cm acc
model = lModels[xvIndex][1]
print(model)
# fit the model
model.fit(X,y)
print('Done ...')


# Save the model
print("\n**** Save the model****")
import pickle
filename = './data/farmingham-model.pkl'
pickle.dump(model,open(filename,'wb'))
print("\n *** Done ***")


# Save vars as dicts

print('\n*** Save vars as dict')
dVars = {}
dVars['clsvars'] = clsVars
dVars['allcols'] = allCols
print(dVars)

# Save dvars

print('\n*** saving dvars ***')
filename = './data/farmingham-dvars.pkl'
pickle.dump(dVars,open(filename,'wb'))
print('\n*** Done ***')
















