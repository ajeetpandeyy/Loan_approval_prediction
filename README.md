# Loan_approval_prediction
Data Science or data analytics is a process of analyzing large data set of data points 
to get answer on questions related to data set.(data point is a discrete unit of information).In a general sense, any single fact is a data point

#Problem Statement:
A Bank is trying to create a Model to help them predict whether to grant loan to a customer.we have been give a training data set consisting of 614 loan applications with information
like the applicants name, personal details, financial information and requested loan amount and related details and the outcome (whether the application was approved or rejected). 
Our task is to create a model to predict the outcome Loan_Status.


STEP 1) Importe Python libraries:
To read data from provided data set and do Mathematical computation and Visualization on provided data.
I have import Pandas,Numpy and seaborn. which makes importing and analyzing data much easier
Numpy arrays and pandas dataframes will help us in manipulating data

--Reading training dataset in a dataframe using Python Pandas library
Train_loan_Df=pd.read_csv("F:/download/data csv/train_LoanStatus.csv") --'Train_loan_Df' is data frame
Train_loan_Df.head(10) --checking  First 10 Rows of training Dataset

STEP 2) Exploratory Data Analysis:
Exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods
primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.
Understanding the various columns of the dataset.

	i)Summary of numerical variables for training data set
			Train_loan_Df.describe()  
		----Dataframe.describe() is used to view some basic statistical details like count, percentile, mean,max, std etc. of a data frame of numeric values.
	    For the non-numerical values like  Property_Area, Credit_History etc., we can look at frequency distribution to understand whether they make sense or not.
		# Get the unique values and their frequency of variable Property_Area

	ii) Anylizing Data by plots:
		# Box Plot for understanding the distributions and to observe the outliers.
		Train_loan_Df.boxplot(column='ApplicantIncome')
		--the above Box Plot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society.
		
		# Box Plot for variable ApplicantIncome by variable Education of training data set
		Train_loan_Df.boxplot(column='ApplicantIncome', by = 'Education')
		--there are a higher number of graduates with very high incomes, which are appearing to be the outliers
		
		#Train_loan_Df.boxplot(column='LoanAmount', by = 'Gender')
		70% of Male has 
		
	--Understanding Distribution of Categorical Variables

		# Loan approval rates in absolute numbers
		loan_approval = Train_loan_Df['Loan_Status'].value_counts()['Y']
		print(loan_approval) 
		--422 number of loans were approved
		
		# Credit History and Loan Status
		pd.crosstab(Train_loan_Df.Loan_Status,Train_loan_Df.Credit_History, normalize=True)*100
			Credit_History	0.0		    1.0
			Loan_Status		
				N	    13.355049	  17.915309
				Y	    1.140065	  67.589577
		
		----67.58% of the applicants whose loans were approved have Credit_History equals to 1.

		pd.crosstab(Train_loan_Df.Married,Train_loan_Df.Loan_Status,normalize='index')*100
		
            Loan_Status 	N		  Y
            Married		
				No		37.089202	62.910798
				Yes		28.391960	71.608040
		--71.60% of the applicants whose loans were approved are Married
		
		pd.crosstab(Train_loan_Df.Dependents,Train_loan_Df.Loan_Status,normalize='index')*100
		
			Loan_Status		N			Y
			Dependents		
				0		31.014493	68.985507
				1		35.294118	64.705882
				2		24.752475	75.247525
				3+		35.294118	64.705882
		--75.24% of the applicants whose loans were approve has 2 Dependents	
				
		pd.crosstab(Train_loan_Df.Education,Train_loan_Df.Loan_Status,normalize='index')*100
			Loan_Status			N			Y
			Education		
			Graduate		29.166667	70.833333
			Not Graduate	38.805970	61.194030
			
		--Graduate client has 10% more chaces to get loan Approval than not gratudate
		
		pd.crosstab(Train_loan_Df.Self_Employed,Train_loan_Df.Loan_Status,normalize='index')		
			Loan_Status			N		   Y
			Self_Employed		
				No			0.314000	0.686000
				Yes			0.317073	0.682927
		--Equavelent chance to get loan for employed or not employed clients
		
		pd.crosstab(Train_loan_Df.Loan_Status,Train_loan_Df.Property_Area,normalize='columns')
		
		Property_Area		Rural	Semiurban	Urban
			Loan_Status			
				N		  0.385475	0.23176		0.341584
				Y   	  0.614525	0.76824		0.658416
		--Chances of getting a loan will be higher for Applicants from Urban Areas		

		
		
	
STEP 3):DATA Cleansing
Training a model with a dataset that has a lot of missing values can drastically impact the machine learning model’s quality. 
Some algorithms such as scikit-learn estimators assume that all values are numerical and have and hold meaningful value.
our strategy would be to impute the missing values
we need to infer those missing values from the existing part of the data. 
Imputation Using (Mean/Median) Values:It can only be used with numeric data.
Imputation Using (Most Frequent) or (Zero/Constant) Values:replacing missing data with the most frequent values within each column: it Works well with categorical features.

--#Data Cleansing by sklearn's SimpleImputer
sklearn requires all inputs to be numeric, we should convert all our categorical variables into numeric by encoding the categories.
Before that we will fill all the missing values in the dataset

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.NAN, strategy='most_frequent', fill_value=None, verbose=0, copy=True)
imp.fit(Train_loan_Df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']])
Train_loan_Df_Im= imp.transform(Train_loan_Df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']])

----Second and simple Approach
#Imputing Missing values with mean for continuous variable

Train_loan_Df['LoanAmount'].fillna(Train_loan_Df['LoanAmount'].mean(), inplace=True)
Train_loan_Df['Loan_Amount_Term'].fillna(Train_loan_Df['Loan_Amount_Term'].mean(), inplace=True)
Train_loan_Df['ApplicantIncome'].fillna(Train_loan_Df['ApplicantIncome'].mean(), inplace=True)
Train_loan_Df['CoapplicantIncome'].fillna(Train_loan_Df['CoapplicantIncome'].mean(), inplace=True)

#Imputing Missing values with mode for categorical variables
--Impute missing values for Gender
Train_loan_Df['Gender'].fillna(Train_loan_Df['Gender'].mode()[0], inplace=True)
--Impute missing values for Married
Train_loan_Df['Married'].fillna(Train_loan_Df['Married'].mode()[0], inplace=True)
Train_loan_Df['Dependents'].fillna(Train_loan_Df['Dependents'].mode()[0], inplace=True)
Train_loan_Df['Loan_Amount_Term'].fillna(Train_loan_Df['Loan_Amount_Term'].mode()[0], inplace=True)
Train_loan_Df['Credit_History'].fillna(Train_loan_Df['Credit_History'].mode()[0], inplace=True)


--Logistic Regression Model:
the logistic regression is a predictive analysis. logistic model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick
--The chances of getting a loan will be higher for:
Applicants having a credit history (we observed this in exploration.)
Applicants with higher applicant and co-applicant incomes
Applicants with higher education level
Properties in urban areas with high growth perspectives

data=Train_loan_Df
cat_vars=['Dependents','Gender','Married','Education','Self_Employed','Property_Area','Credit_History']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg,5)
data=data[to_keep].dropna()
data_X=data.drop(['Loan_Status','Loan_ID'],axis=1)
data_y=data['Loan_Status']

from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=0)
--X_train, y_train are training data &  X_test, y_test belongs to the test dataset.
-- test_size=0.3 it means test sets will be 30% of whole dataset  & training dataset’s size will be 70% of the entire dataset.
-- train_test_split() method will help us by splitting data into train & test set.
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on Train test data set: {:.2f}'.format(logreg.score(X_test, y_test)*100),'%')
--Accuracy of logistic regression classifier on Train test data set: 82.18 %

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

--confusion_matrix:

[[ 28  22]
 [  9 115]]
 print(classification_report(y_test, y_pred))
 
               precision    recall  f1-score   support

           N       0.78      0.56      0.65        50
           Y       0.84      0.94      0.89       124

    accuracy                           0.83       174
   macro avg       0.81      0.75      0.77       174
weighted avg       0.82      0.83      0.82       174
 
 data_X['total_income']=data_X['ApplicantIncome']+data_X['CoapplicantIncome']
 
 Train_loan_Df['total_income']=Train_loan_Df['ApplicantIncome']+Train_loan_Df['CoapplicantIncome']
take_loan=list(Train_loan_Df[Train_loan_Df['Loan_Status']=='Y']['total_income'])
not_take_loan=list(Train_loan_Df[Train_loan_Df['Loan_Status']=='Y']['total_income'])

X_train, X_test, y_train, y_test = train_test_split(data_X.drop(['ApplicantIncome','CoapplicantIncome'],axis=1), data_y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)*100),'%')
--Accuracy of logistic regression classifier on test set: 82.76 %

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))

--confusion_matrix
[[ 28  22]
 [  8 116]]
 
              precision    recall  f1-score   support

           N       0.78      0.56      0.65        50
           Y       0.84      0.94      0.89       124

    accuracy                           0.83       174
   macro avg       0.81      0.75      0.77       174
weighted avg       0.82      0.83      0.82       174

from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

--Decsion Tree

Now we fit Decision tree algorithm on training data, predicting labels for validation dataset and printing the accuracy of the model using various parameters.

from sklearn.tree import DecisionTreeClassifier  
--DecisionTreeClassifier(): This is the classifier function for DecisionTree. It is the main function for implementing the algorithms. 
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=0)
Dtree = DecisionTreeClassifier(criterion = 'entropy').fit(X_train,y_train)
y_pred = Dtree.predict(X_test)
print("The prediction accuracy is: ",Dtree.score(X_test,y_test)*100,"%")
--The prediction accuracy is:  77.58620689655173 %

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
--confusion_matrix
[[ 32  18]
 [ 21 103]]
              precision    recall  f1-score   support

           N       0.60      0.64      0.62        50
           Y       0.85      0.83      0.84       124

    accuracy                           0.78       174
   macro avg       0.73      0.74      0.73       174
weighted avg       0.78      0.78      0.78       174


 --Accuracy of logistic regression classifier on Train test data set: 82.18 %
 --Accuracy of logistic regression classifier on test set: 82.76 %
 --The prediction accuracy is:  77.58620689655173 %
