import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, stdev
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

diabetes = pd.read_csv("diabetes.csv")
diabetes.head()
diabetes.info()
diabetes.describe()
diabetes.drop_duplicates()
#first store the features in a seperate dataframe.
features = diabetes.drop("Outcome",axis = 1).copy()
#Now plot a boxplot to identify the outliers in our features.
sns.boxplot(data = features, orient = 'h', palette = 'Set3', linewidth = 2.5 )
plt.title("Features Box Plot")
sns.boxplot(x = diabetes["Outcome"], orient = 'h', linewidth = 2.5 )
plt.title("Target Column Box Plot")

from scipy import stats
def removeoutliers(df=None, columns=None):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        floor, ceil = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[column] = df[column].clip(floor, ceil)
        print("The columnn: {column}, has been treated for outliers.\n")
    return df


diabetes = removeoutliers(diabetes,[col for col in features.columns])
sns.boxplot(data = diabetes, orient = 'h', palette = 'Set3', linewidth = 2.5 )
plt.title("Box Plot after treating outliers")
diabetes.hist(bins = 10, figsize = (10,10))

sns.countplot('Outcome',data=diabetes)
print('Outcome Class Ratio:',sum(diabetes['Outcome'])/len(diabetes['Outcome']))
#plot the heatmap
sns.heatmap(diabetes.corr())
y = diabetes.loc[:,'Outcome']
X = diabetes.drop(['Outcome'],axis = 1).copy()
X.head()
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


def train_model(model_name,model):
  
    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    accuracy = []
  
    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = scaled_X[train_index], scaled_X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model.fit(x_train_fold, y_train_fold)
        accuracy.append(model.score(x_test_fold, y_test_fold))
  
    # Print the output.
    #   print(f'The model {model_name} has an Average Accuracy:',round(mean(accuracy)*100,2), '%')
    # print('\nMaximum Accuracy that can be obtained:',round(max(accuracy)*100,2), '%')
    # print('\nStandard Deviation:', stdev(accuracy))
    # print('\n\n\n')
models = {}
models["'Logistic Regression'"] = LogisticRegression(random_state = 12345)
models["'K Nearest Neighbour'"] = KNeighborsClassifier()
models["'Decision Tree'"] = DecisionTreeClassifier(random_state = 12345)
models["'Random Forest'"] = RandomForestClassifier(random_state = 12345)
models["'SVM'"] = SVC(gamma='auto', random_state = 12345)
models["'XGB'"] = GradientBoostingClassifier(random_state = 12345)

for key, values in models.items():
    train_model(key,values)
plt.show()
plt.show()