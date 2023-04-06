# -*- coding: utf-8 -*-
"""
Created: 21-Aug-2015
Updated: 17-Feb-2023
@filename: utils.py
@describe: utility functions
@dataset: Nil
@author: cyruslentin
"""
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
import json

# function 
# return node text or None 
def getValueOfNode(node):
    return node.text if node is not None else None

# function read xml file
# limitations - should have single leaf or singe node
"""
sample xml record structure
<?xml version = "1.0"?>
<Friends>
	<Record>
		<Display>Tanmay Patil</Display>
		<Name>
			<FirstName>Tanmay</FirstName>
			<LastName>Patil</LastName>
		</Name>	
		<Age>25</Age>
		<Contact>
			<Mobile>9876543210</Mobile>
			<Email>tanmaypatil@xyz.com</Email>
		</Contact>
		<Address>
			<City>Bangalore</City>
			<State>Karnataka</State>
			<Pin>560212</Pin>
		</Address>
	</Record>
</Friends>
"""
def read_xml(fileName):
    #fileName = './data/Friends.xml'
    # create empty data frame
    df = pd.DataFrame()
    # filename with math from cwd
    oTree = et.parse(fileName)
    # get root
    oRoot = oTree.getroot()
    #print(oRoot.tag)
    # get records from root
    for oRecord in oRoot:
        #print(oRecord.tag)
        recDict={}
        for oField in oRecord:
            #print(oField.tag)
            vFieldValue = getValueOfNode(oField)
            #print(type(vFieldValue))
            if (vFieldValue.strip() != ''):
                recDict[oField.tag] = vFieldValue
            else:
                fldDict={}
                for oSubField in oField:
                    #print(oSubField.tag,':',getvalueofnode(oSubField))
                    fldDict[oSubField.tag] = getValueOfNode(oSubField)
                #print(fldDict)    
                recDict.update(fldDict)
        print(recDict)
        dft= pd.DataFrame(recDict, index=[0])
        df = df.append(dft, ignore_index=True)
    # return
    return(df)    

# function read json file 
# limitations - should have single leaf or singe node
"""
sample json record structure
{
"Friends":[
   {
	  "Display":"Tanmay Patil",
      "Name": {
         "FirstName": "Tanmay",
         "LastName": "Patil"
      },
	  "Age":"25",
      "Contact": {
         "Mobile": "9876543210",
         "Email": "tanmaypatil@xyz.com"
      },
      "Address": {
         "City": "Bangalore",
         "State": "Karnataka",
         "Pin": "560212"
      }
   }
]}
"""
def read_json(fileName):
    #fileName = './data/Friends.json'
    # create empty dataframe
    df = pd.DataFrame()
    # read json file
    with open(fileName, 'rb') as jsonFile:  
        data = json.load(jsonFile)
        #print(data)
        #print(data.keys())
        # get json Name (top level name / table name)
        for jsonName in data.keys():
            #print(jsonName)
            # get record dict
            for recData in data[jsonName]:
                recDict = {}
                for k, v in recData.items():
                    if not isinstance(v, dict):
                        recDict[k] = v
                    else:
                        recDict.update(v)
                print("\n"+str(recDict))        
                dft= pd.DataFrame(recDict, index=[0])
                df = df.append(dft, ignore_index=True)
    # return
    return(df)



# space count per coulmn
"""
returns: 
    number of rows which contain <blank>
usage: 
    colSpaceCount(colName)
""" 
def colSpaceCount(colName):
    return (colName.str.strip().values == '').sum()


# space count for data frame
"""
returns:  
    number of rows which contain <blank> iterating through each col of df
usage: 
    SpaceCount(df)
"""
def SpaceCount(df): 
    colNames = df.columns
    dsRetValue = pd.Series() 
    for colName in colNames:
        if df[colName].dtype == "object": 
            dsRetValue[colName] = colSpaceCount(df[colName])
    return(dsRetValue)


# outlier limits
"""
returns: 
    upper boud & lower bound for array values or df[col] 
usage: 
    OutlierLimits(df[col]): 
"""
def colOutlierLimits(colValues, pMul=3): 
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    pMul = float(pMul)    
    q1, q3 = np.percentile(colValues, [25, 75])
    iqr = q3 - q1
    ll = q1 - (iqr * pMul)
    ul = q3 + (iqr * pMul)
    return ll, ul


# outlier count for column
"""
returns: 
    count of outliers in the colName
usage: 
    colOutCount(colValues)
"""
def colOutlierCount(colValues, pMul=3):
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    ll, ul = colOutlierLimits(colValues, pMul)
    ndOutData = np.where((colValues > ul) | (colValues < ll))
    ndOutData = np.array(ndOutData)
    return ndOutData.size


# outlier count for dataframe
"""
returns: 
    count of outliers in each column of dataframe
usage: 
    OutlierCount(df): 
"""
def OutlierCount(df, pMul=3): 
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    pMul = float(pMul)    
    colNames = df.columns
    dsRetValue = pd.Series() 
    for colName in colNames:
        #print(df[colName].dtypes)
        if (df[colName].dtypes == 'object' or df[colName].dtypes == 'bool'):
            continue
        #print(colName)
        colValues = df[colName].values
        #print(colValues)
        #outCount = colOutCount(colValues)
        #print(outCount)
        dsRetValue[colName] = colOutlierCount(colValues, pMul)
    return(dsRetValue)


# oulier index for column
"""
returns: 
    row index in the colName
usage: 
    colOutIndex(colValues)
"""
def colOutlierIndex(colValues, pMul=3):
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    ll, ul = colOutlierLimits(colValues, pMul)
    ndOutData = np.where((colValues > ul) | (colValues < ll))
    ndOutData = np.array(ndOutData)
    return ndOutData


# oulier index for data frame
"""
returns: 
    row index of outliers in each column of dataframe
usage: 
    OutlierIndex(df): 
"""
def OutlierIndex(df, pMul=3): 
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    pMul = float(pMul)    
    colNames = df.columns
    dsRetValue = pd.Series() 
    for colName in colNames:
        if (df[colName].dtypes == 'object' or df[colName].dtypes == 'bool'):
            continue
        colValues = df[colName].values
        dsRetValue[colName] = str(colOutlierIndex(colValues, pMul))
    return(dsRetValue)


# outlier values for column 
"""
returns: 
    actual outliers values in the colName
usage: 
    colOutValues(colValues)
"""
def colOutlierValues(colValues, pMul=3):
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    ll, ul = colOutlierLimits(colValues, pMul)
    ndOutData = np.where((colValues > ul) | (colValues < ll))
    ndOutData = np.array(colValues[ndOutData])
    #ndOutData = np.array(ndOutData)
    return ndOutData


# outlier values for dataframe 
"""
returns: 
    actual of outliers in each column of dataframe
usage: 
    OutlierValues(df): 
"""
def OutlierValues(df, pMul=3): 
    if (pMul != 3 and pMul != 2.5 and pMul != 2 and pMul != 1.5):
        pMul = 3
    pMul = float(pMul)    
    colNames = df.columns
    dsRetValue = pd.Series() 
    for colName in colNames:
        if (df[colName].dtypes == 'object' or df[colName].dtypes == 'bool'):
            continue
        colValues = df[colName].values
        dsRetValue[colName] = str(colOutlierValues(colValues, pMul))
    return(dsRetValue)


# column level handle outlier by capping
# at lower limit & upper timit respectively
"""
returns: 
    array values or df[col].values without any outliers
usage: 
    HandleOutlier(df[col].values): 
"""
def colHandleOutliers(colValues, pMul=3):
    ll, ul = colOutlierLimits(colValues, pMul)
    colValues = np.where(colValues < ll, ll, colValues)
    colValues = np.where(colValues > ul, ul, colValues)
    return (colValues)


# data frame level handline outliers
"""
desc:
    HandleOutliers - removes Outliers from all cols in df except exclCols 
usage: 
    HandleOutliers(df, colClass) 
params:
    df datarame, exclCols - col to ignore while transformation, Multiplier  
"""
def HandleOutliers(df,  lExclCols=[], pMul=3):
    #lExclCols = depVars
    # preparing for standadrising
    # orig col names
    colNames = df.columns.tolist()
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle outlier for each col
    for colName in colNames:
        colType =  df[colName].dtype  
        df[colName] = colHandleOutliers(df[colName], pMul)
        if df[colName].isnull().sum() > 0:
            df[colName] = df[colName].astype(np.float64)
        else:
            df[colName] = df[colName].astype(colType)    
    return df


# data frame handle zeros replace with null
"""
desc:
    HandleZeros - removes zeros from all cols specifie in df 
usage: 
    HandleZeros(df, colClass) 
params:
    df datarame, exclCols - col to ignore while transformation, Multiplier  
"""
def HandleZeros(df, lZeroCols=[]):
    # if not list convert to list
    if not isinstance(lZeroCols, list):
        lZeroCols = [lZeroCols]
    #print(lExclCols)
    # handle outlier for each col
    for colName in lZeroCols:
        if ((df[colName]==0).sum() > 0):
            df[colName] = np.where(df[colName] == 0, None, df[colName])
            df[colName] = df[colName].astype(float)
    return df


# data frame handle nulls replace with ReplVals of the columns as per replBy vars
"""
desc:
    HandleNulls - removes Outliers from all cols in df except exclCols 
usage: 
    HandleNulls(df, replBy, colClass) 
params:
    df datarame, 
    replBy - mean, median, minimum (of mean & median), maximum (of mean & median) 
    exclCols - col to ignore while transformation, Multiplier  
"""
def HandleNulls(df, replBy, lExclCols=[]):
    #lExclCols = depVars
    # preparing for standadrising
    # orig col names
    colNames = df.columns.tolist()
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle outlier for each col
    for colName in colNames:
        if ((df[colName].isnull()).sum() > 0):
            if (replBy == "mean"):
                replVals = df[colName].mean()
            elif (replBy == "median"):
                replVals = df[colName].median()
            elif (replBy == "minimum"):
                replVals = min(df[colName].mean(),df[colName].median())
            elif (replBy == "maximum"):
                replVals = max(df[colName].mean(),df[colName].median())
            # replace
            df[colName] = df[colName].fillna(replVals)
    return df

# data frame handle nulls replace with mean of the columns 
def HandleNullsWithMean(df, lExclCols=[]):
    df = HandleNulls(df, "mean", lExclCols)
    return df

# data frame handle nulls replace with median of the columns 
def HandleNullsWithMedian(df, lExclCols=[]):
    df = HandleNulls(df, "median", lExclCols)
    return df

# data frame handle nulls replace with min(mean,median) of the columns 
def HandleNullsWithMinOfMM(df, lExclCols=[]):
    df = HandleNulls(df, "minimum", lExclCols)
    return df

# data frame handle nulls replace with max(mean,median) of the columns 
def HandleNullsWithMaxOfMM(df, lExclCols=[]):
    df = HandleNulls(df, "maximum", lExclCols)
    return df

# identify columns where all value are same
# works only where all cols are numeric
def SameValuesColsNumeric(df, lExclCols=[], Percent=1, Verbose = False):
    # currently can check only 100% same values so Percent has to 1 (100%)
    if (Percent!=1):
        Percent=1
    #lExclCols = depVars
    # preparing for standadrising
    # orig col names
    colNames = df.columns.tolist()
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle same value for each col
    lRetVals = []
    dsRetValue = pd.Series() 
    for colName in colNames:
        colVrnc = df[colName].var()
        # print(colName)
        # print(colVrnc)
        # print("")
        dsRetValue[colName] = '%.2f' % colVrnc
        if colVrnc == Percent:
            lRetVals.append(colName)
    if (Verbose):       
        print(dsRetValue)    
    return lRetVals

# identify columns where all value are same (uniqVals=1)
# identify columns where two value are unique (uniqVals=2)
# identify columns where two value are unique (uniqVals=3)
# works only where all cols are numeric
def SameValuesCols(df, lExclCols=[], uniqVals=1, Verbose = False):
    #lExclCols = depVars
    # preparing for standadrising
    # orig col names
    colNames = df.columns.tolist()
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle same value for each col
    lRetVals = []
    dsRetValue = pd.Series() 
    for colName in colNames:
        cntUniq = df[colName].nunique()
        #cntRecs  = len(df.index)
        dsRetValue[colName] = '%7d' % cntUniq
        if (cntUniq <= uniqVals):
            lRetVals.append(colName)
    if (Verbose):       
        print(dsRetValue)    
    return lRetVals


# identify columns with more than 100% unique values
def UniqValuesCols(df, lExclCols=[], Percent=0.95, Verbose = False):
    #lExclCols = depVars
    # preparing for standadrising
    # orig col names
    colNames = df.columns.tolist()
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle uniq values for each col
    dsRetValue = pd.Series() 
    lRetVals = []
    for colName in colNames:
        cntUniq = df[colName].nunique()
        cntRecs  = len(df.index)
        perRecs  = cntUniq / cntRecs
        dsRetValue[colName] = '%.2f' % perRecs
        if perRecs >= Percent:
            lRetVals.append(colName)
    if (Verbose):       
        print(dsRetValue)    
    return lRetVals

# identify columns with more than 50% null values
def NullValuesCols(df, lExclCols=[], Percent=0.5, Verbose = False):
    #lExclCols = depVars
    # preparing for standadrising
    # orig col names
    colNames = df.columns.tolist()
    #Percent=0.5
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            colNames.remove(vExclCol)
    # handle null values for each col
    dsRetValue = pd.Series() 
    lRetVals = []
    for colName in colNames:
        cntNulls = df[colName].isnull().sum()
        cntRecs  = len(df.index)
        perRecs  = cntNulls / cntRecs
        dsRetValue[colName] = '%.2f' % perRecs
        if perRecs >= Percent:
            lRetVals.append(colName)
    if (Verbose):       
        print(dsRetValue)    
    return (lRetVals)


# get multi colinearity columns
"""
returns: 
    list of cols to be dropped because the are co-linear
usage: 
    MulCorrCols(dfc, depVars, bVerbose, ExclCols)
    dfc should be generated by df.corr()
    depVars : the dependent variable for linear regression model
	bVerbose shows the working of the function
    ExclCols list of cols to exclude from the process 
"""
def MulCorrCols(dfc, depVars, bVerbose=False, vMaxCor=0.9, lExclCols=[]):
    # remove depVars from col
    dfc = dfc.drop(depVars, axis=1)
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    #print(lExclCols)
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        for vExclCol in lExclCols:
            dfc = dfc.drop(vExclCol, axis=1)
            dfc = dfc[dfc.index!=vExclCol]
    # dfdepVars
    dfv = dfc[dfc.index==depVars]
    # col names
    colNames = dfc.columns.tolist()
    # col count 
    vColCount = len(colNames)
    # list of cols to drop
    dropCols = []
    # loop from 0 to less than colCount
    for i in range(0,vColCount):
        if bVerbose:
            print("\ni = ",i,"******************")
        iCol = colNames[i]
        # loop from i+1 to less than colCount
        for j in range(i+1,vColCount):
            if bVerbose:
                print("\nj = ",j)
            jCol = colNames[j]
            # print iColName & jColName
            if bVerbose:
                print(iCol,"&",jCol)
            # get corr from dfc for iColName & jColName
            vCor = dfc.iloc[i, j]     
            if bVerbose:
                print("Correl:",vCor)
            # if corr bet two cols > vMaxCor
            if abs(vCor) > vMaxCor:
                # get corr of col & depVars
                iCor = dfv.iloc[0,i]
                jCor = dfv.iloc[0,j]
                if bVerbose:
                    #print("")
                    #print(i,j)
                    #print(iCol,jCol)
                    #print(vCor)
                    print("Corr Of",iCol," with DV:",iCor)
                    print("Corr of",jCol," with DV:",jCor)
                # which ever corr of depVars is lower 
                # apped in list of Col to Drop
                if abs(iCor) > abs(jCor):
                    if jCol not in dropCols:
                        dropCols.append(jCol)
                    if bVerbose:
                        print(jCol)
                else:
                    if iCol not in dropCols:
                        dropCols.append(iCol)
                    if bVerbose:
                        print(iCol)
    return (dropCols)                


# standardize data
"""
desc:
    standardize data - all cols of df will be Standardized except colClass 
    x_scaled = (x — mean(x)) / stddev(x)
    all values will be between 1 & -1
usage: 
    StandardizeData(df, colClass) 
params:
    df datarame, colClass - col to ignore while transformation  
"""
def StandardizeData(df, lExclCols=[]):
    # preparing for standadrising
    # orig col names
    colNames = df.columns.tolist()
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        dfExcelCols = df[lExclCols]
    # standardizaion : 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # fit
    ar = scaler.fit_transform(df)
    # transform
    df = pd.DataFrame(data=ar)
    # rename back to orig cols
    df.columns = colNames
    # overwrite ExclCols with dataframe of ExclCols
    if lExclCols != []:
        df[lExclCols] = dfExcelCols 
    return(df)


# normalize data
"""
desc:
    normalize data - all cols of df will be Normalized except lExclCols
    x_scaled = (x-min(x)) / (max(x)–min(x))
    all values will be between 0 & 1
usage: 
    NormalizeeData(df, colClass) 
params:
    df datarame, lExclCols - cols to ignore while transformation  
"""
def NormalizeData(df, lExclCols=[]):
    # preparing for normalising
    # orig col names
    colNames = df.columns.tolist()
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        dfExcelCols = df[lExclCols]
    from sklearn.preprocessing import MinMaxScaler
    # normalizing the data
    scaler = MinMaxScaler()
    # fit
    ar = scaler.fit_transform(df)
    # transform
    df = pd.DataFrame(data=ar)
    # rename back to orig cols
    df.columns = colNames
    # overwrite ExclCols with dataframe of ExclCols
    if lExclCols != []:
        df[lExclCols] = dfExcelCols 
    return(df)


# Max Abs Scalaed Data
"""
desc:
    MaxAbsScaled data - all cols of df will be MaxAbsScaled except colClass 
    x_scaled = x / max(abs(x))
Usage: 
    MaxAbsScaledData(df, colClass) 
Params:
    df datarame, colClass - col to ignore while transformation  
"""
def MaxAbsScaledData(df, lExclCols=[]):
    # preparing for MaxAbsScalar
    # orig col names
    colNames = df.columns.tolist()
    # if not list convert to list
    if not isinstance(lExclCols, list):
        lExclCols = [lExclCols]
    # if not empty, create a dataframe of ExclCols
    if lExclCols != []:
        dfExcelCols = df[lExclCols]
   # MaxAbsScalar the data 
    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()
    # fit
    ar = scaler.fit_transform(df)
    # transform
    df = pd.DataFrame(data=ar)
    # rename back to orig cols
    df.columns = colNames
    # overwrite ExclCols with dataframe of ExclCols
    if lExclCols != []:
        df[lExclCols] = dfExcelCols 
    return(df)


# getFeatureScoresXTC - Extra Tree Classifier
"""
desc:
    prints feature scores of all cols except colClass 
usage: 
    getFeatureScoresXTC(df, colClass) 
params:
    df datarame, colClass - col to ignore while transformation  
"""
def getFeatureScoresXTC(df, colClass):
    # make into array
    #print("\n*** Prepare Data ***")
    # store class variable  ... change as required
    clsVars = colClass
    allCols = df.columns.tolist()
    #print(allCols)
    allCols.remove(clsVars)
    #print(allCols)
    # split into X & y        
    X = df[allCols].values
    y = df[clsVars].values

    # feature extraction with ExtraTreesClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    # extraction
    model = ExtraTreesClassifier(n_estimators=10, random_state=707)
    model.fit(X, y)
    #print("\n*** Column Scores ***")
    # summarize scores
    np.set_printoptions(precision=3)
    #print(model.feature_importances_)
    # data frame
    dfm =  pd.DataFrame({'Cols':allCols, 'Imp':model.feature_importances_})  
    dfm.sort_values(by='Imp', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
    return (dfm)


# getFeatureScoresSKB - Select K Best
"""
desc:
    prints feature scores of all cols except colClass 
usage: 
    getFeatureScoresXTC(df, colClass) 
params:
    df datarame, colClass - col to ignore while transformation  
"""
def getFeatureScoresSKB(df, colClass):
    # make into array
    #print("\n*** Prepare Data ***")
    # store class variable  ... change as required
    clsVars = colClass
    allCols = df.columns.tolist()
    #print(allCols)
    allCols.remove(clsVars)
    #print(allCols)
    # split into X & y        
    X = df[allCols].values
    y = df[clsVars].values
    
    # Feature extraction with selectBest
    np.random.seed(707)
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    # feature extraction
    model = SelectKBest(score_func=f_classif, k=4)
    fit = model.fit(X, y)
    # summarize scores
    np.set_printoptions(precision=3)
    #print(fit.scores_)
    # data frame
    dfm =  pd.DataFrame({'Cols':allCols, 'Imp':fit.scores_})  
    dfm.sort_values(by='Imp', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
    return (dfm)


# select feature to drop - chi square
"""
desc:
    select features or columns to drop based on chi square alog
usage: 
    selectFeaturesToDrop(df, clsVars, vColDrop) 
params:
    df datarame, 
    colClass - col to ignore while transformation, 
    nColsDrop - number of cols to drop
"""
def DropFeaturesCh2(df, colClass, nColsDrop):
    # make into array
    #print("\n*** Prepare Data ***")
    # store class variable  ... change as required
    clsVars = colClass
    allCols = df.columns.tolist()
    #print(allCols)
    allCols.remove(clsVars)
    vColRetain = len(allCols)-nColsDrop
    #print(allCols)
    # split into X & y        
    X = df[allCols].values
    y = df[clsVars].values
    from sklearn.feature_selection import SelectKBest, chi2
    model = SelectKBest(chi2, k=vColRetain)
    fit = model.fit(X, y)
    chkCols = model.get_support().tolist()
    print(allCols)
    print(chkCols)
    #lDropCols = set(allCols)^set(chkCols)
    lDropCols = []
    for i in range(0,len(chkCols)):
        if (chkCols[i]==False):
            lDropCols.append(allCols[i])
    return (lDropCols)


# select feature to drop - worst of combained rank
"""
desc:
    select features or columns to drop based on feature selection scrore / rank 
usage: 
    selectFeaturesToDrop(df, clsVars, vColsDrop) 
params:
    df datarame, 
    colClass - col to ignore while transformation, 
    vColsDrop - number of cols to drop
"""
def DropFeaturesRank(df, colClass, nColsDrop):
    # get feature scores
    dfSX = getFeatureScoresXTC(df, colClass)
    dfSY = getFeatureScoresSKB(df, colClass)
    # add Rank Col
    dfSX['Rank'] = range(1,len(dfSX.index)+1)
    dfSY['Rank'] = range(1,len(dfSY.index)+1)
    # rename cols so not duplicate when dfs are merged
    dfSX.columns = ['Cols', 'ImpX', 'RankX']
    dfSY.columns = ['Cols', 'ImpY', 'RankY']
    # merge dfs 
    dfFS = pd.merge(dfSX, dfSY, on='Cols', how='inner')
    # combined rank column
    dfFS['Rank'] = dfFS['RankX'] + dfFS['RankY']
    # sort descending based on combined rank
    dfFS = dfFS.sort_values(by=['Rank'], ascending=False)
    dfFS = dfFS.reset_index() 
    print(dfFS)
    # get list of cols to be dropped
    lDropCols = dfFS['Cols'][0:nColsDrop]
    lDropCols = lDropCols.tolist()
    # return
    return (lDropCols)



# get OverSampleData
"""
install:
    !pip install -U imbalanced-learn
url:
    https://pypi.org/project/imbalanced-learn/
desc:
    Random Over Sampler ... 
    creates duplicate records of the lower sample
    to match the sample size of highest size class
usage: 
    getOverSamplerData(X, y) ... requires standard X, y 
"""
def getOverSamplerData(X,y): 
    # import
    from imblearn.over_sampling import RandomOverSampler
    # create os object
    os =  RandomOverSampler(random_state = 707)
    # generate over sampled X, y
    return (os.fit_resample(X, y))


# get SMOTE Sampler Data
"""
install:
    !pip install -U imbalanced-learn
url:
    https://pypi.org/project/imbalanced-learn/
desc:
    SMOTE - Synthetic Minority Oversampling Technique 
    creates random new synthetic records
    to match the sample size of highest size class
usage: 
    getSmoteSamplerData(X, y) ... requires standard X, y 
"""
def getSmoteSamplerData(X,y): 
    # import
    from imblearn.over_sampling import SMOTE
    # create smote object
    sm = SMOTE(random_state = 707)
    # generate over sampled X, y
    return (sm.fit_resample(X, y))


# get UnderSamplerData
"""
install:
    !pip install -U imbalanced-learn
url:
    https://pypi.org/project/imbalanced-learn/
desc:
    Random Under Sampler ... 
    deletes records of the higher sample
    to match the sample size of lowest size class
usage:  
    getUnderSamplerData(X, y)
params:
    requires standard X, y 
"""
def getUnderSamplerData(X,y): 
    # import
    from imblearn.under_sampling import RandomUnderSampler
    # create os object
    us =  RandomUnderSampler(random_state = 707, replacement=True)
    # generate over sampled X, y
    return (us.fit_resample(X, y))


# one hot encoding
"""
desc:
    One Hot Encoding 
    Col With Categoric Values A & B is converted to ColA & ColB with 0s & 1s
usage: 
    oheBind(pdf, encCol)
params:
    pdf - data frame, encCol - column to be encoded
returns:
    df with oheCols & encCol deleted
"""
def oheBind(pdf, encCol):
    ohe = pd.get_dummies(pdf[[encCol]])
    #ohe.columns = pdf[encCol].unique()
    rdf = pd.concat([pdf, ohe], axis=1)
    rdf = rdf.drop(encCol, axis=1)
    return(rdf)


##############################################################
# Step Regression Function
# https://www.investopedia.com/terms/s/stepwise-regression.asp
##############################################################
import statsmodels.api as sm
def StepRegression(X, y,
                       initial_list=[], 
                       threshold_out = 0.05, 
                       verbose=True):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_out:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add ' + best_feature +' with p-value ' + str(best_pval))
        if not changed:
            break
    return included

# get cleanData - Function
# removes all chars specifed in lChars string
# by default all punctuation
import string as st
def getCleanData(pData, lChars=st.punctuation):
    # print arg
    #print(pName)
    # clean
    vRetVals = pData.translate(str.maketrans(" ", " ", lChars))
    vRetVals = vRetVals.strip()
    return(vRetVals)  


# relative root mean squared error
"""
desc:
    relative root mean squared error
usage: 
    def getRRMSE(true, pred):
params:
    true - actuals values in the series
    pred - predicted values
returns:
    continous numeric value
    giving rrmse
    # Excellent when RRMSE < .1
    # Good when RRMSE is between .1 and .2
    # Fair when RRMSE is between .2 and .5
    # Poor when RRMSE > .5
"""
def getRRMSE(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse = np.sqrt(squared_error)
    return rrmse


# generic lookup function
"""
desc:
    get lookup value 

usage: 
    getLookupValue(df, pLookVals, pLookCols, pRetsIndx)
params:
    pdf - source or lookup data frame 
    pLookVals - Lookup Value (the key)
    pLookCols - Lookup Column in the lookup data frame
    pRetsIndx - Column index number of the return value
returns:
    df with oheCols & encCol deleted
"""
def getLookupValue(pdf, pLookVals, pLookCols, pRetsIndx):
    # pdf = dfa
    # pLookVals = "BM856"
    # pLookCols = "AuthID"
    # pRetsCols = 1
    # print("Lookup Value :",pLookVals)
    # print("Lookup Column:",pLookCols)
    # print("Return Column:",pRetsIndx)
    # lookup
    try:
        dfs = pdf[ pdf[pLookCols]==pLookVals ]
        #print(dfs)
        vRetsVals = dfs.iloc[0,pRetsIndx]
        #print(vRetsVals)
    except:
        vRetsVals = None
    return(vRetsVals)        


