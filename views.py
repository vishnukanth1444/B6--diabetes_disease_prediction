
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse


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


# Create your views here.
from Remote_User.models import ClientRegister_Model,diabetes_disease_model,diabetes_disease_prediction,detection_results_model,detection_ratio_model


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_results_model.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def Find_Diabetic_Type_Ratio(request):
    detection_ratio_model.objects.all().delete()
    ratio = ""
    kword = 'Type2'
    print(kword)
    obj = diabetes_disease_prediction.objects.all().filter(Q(Prediction=kword))
    obj1 = diabetes_disease_prediction.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio_model.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Type1'
    print(kword1)
    obj1 = diabetes_disease_prediction.objects.all().filter(Q(Prediction=kword1))
    obj11 = diabetes_disease_prediction.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio_model.objects.create(names=kword1, ratio=ratio1)

    ratio12 = ""
    kword12 = 'No Diabetic'
    print(kword12)
    obj12 = diabetes_disease_prediction.objects.all().filter(Q(Prediction=kword12))
    obj112 = diabetes_disease_prediction.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        detection_ratio_model.objects.create(names=kword12, ratio=ratio12)

    ratio123 = ""
    kword123 = 'Low Diabetic'
    print(kword123)
    obj123 = diabetes_disease_prediction.objects.all().filter(Q(Prediction=kword123))
    obj1123 = diabetes_disease_prediction.objects.all()
    count123 = obj123.count();
    count1123 = obj1123.count();
    ratio123 = (count123 / count1123) * 100
    if ratio123 != 0:
        detection_ratio_model.objects.create(names=kword123, ratio=ratio123)

    ratio1234 = ""
    kword1234 = 'Very Low Diabetic'
    print(kword1234)
    obj1234 = diabetes_disease_prediction.objects.all().filter(Q(Prediction=kword1234))
    obj11234 = diabetes_disease_prediction.objects.all()
    count1234 = obj1234.count();
    count11234 = obj11234.count();
    ratio1234 = (count1234 / count11234) * 100
    if ratio1234 != 0:
        detection_ratio_model.objects.create(names=kword1234, ratio=ratio1234)

    obj = detection_ratio_model.objects.all()
    return render(request, 'SProvider/Find_Diabetic_Type_Ratio.html', {'objs': obj})

def View_Diabetic_Emergency_Details(request):

    keyword="Emergency"

    obj = diabetes_disease_prediction.objects.all().filter(Status=keyword)
    return render(request, 'SProvider/View_Diabetic_Emergency_Details.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = diabetes_disease_prediction.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})


def charts(request,chart_type):
    chart1 = detection_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_results_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def Find_Diabetic_Status_Details(request):

    status=''
    type=''
    obj1 =diabetes_disease_model.objects.values('Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
    )

    diabetes_disease_prediction.objects.all().delete()
    for t in obj1:

        Pregnancies= t['Pregnancies']
        Glucose= t['Glucose']
        BloodPressure= t['BloodPressure']
        SkinThickness= t['SkinThickness']
        Insulin= t['Insulin']
        BMI= t['BMI']
        DiabetesPedigreeFunction= t['DiabetesPedigreeFunction']
        Age= t['Age']

        Glucose1=int(Glucose)

        if Glucose1 >= 160 and Glucose1<= 400:
            status = "Emergency"
            type="Type2"
        elif Glucose1 <= 160 and Glucose1 >=130:
            status = "Diabetic Medication"
            type = "Type1"
        elif Glucose1 <= 130 and Glucose1>70:
            status = "Normal"
            type = "No Diabetic"
        elif Glucose1 <= 70 and Glucose1 >= 60:
            status = "Diabetic Medication"
            type = "Low Diabetic"
        elif Glucose1 <= 60:
            status = "Emergency"
            type = "Very Low Diabetic"

        diabetes_disease_prediction.objects.create(Pregnancies=Pregnancies,
        Glucose=Glucose,
        BloodPressure=BloodPressure,
        SkinThickness=SkinThickness,
        Insulin=Insulin,
        BMI=BMI,
        DiabetesPedigreeFunction=DiabetesPedigreeFunction,
        Age=Age,
        Prediction=type,
        Status=status
            )

    obj =diabetes_disease_prediction.objects.all()
    return render(request, 'SProvider/Find_Diabetic_Status_Details.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_results_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = diabetes_disease_prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.Pregnancies, font_style)
        ws.write(row_num, 1, my_row.Glucose, font_style)
        ws.write(row_num, 2, my_row.BloodPressure, font_style)
        ws.write(row_num, 3, my_row.SkinThickness, font_style)
        ws.write(row_num, 4, my_row.Insulin, font_style)
        ws.write(row_num, 5, my_row.BMI, font_style)
        ws.write(row_num, 6, my_row.DiabetesPedigreeFunction, font_style)
        ws.write(row_num, 7, my_row.Age, font_style)
        ws.write(row_num, 8, my_row.Prediction, font_style)
        ws.write(row_num, 9, my_row.Status, font_style)


    wb.save(response)
    return response


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

diabetes = pd.read_csv("./diabetes.csv")
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

def train_model1(model_name,model):

    obj=''
    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    accuracy = []

    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = scaled_X[train_index], scaled_X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model.fit(x_train_fold, y_train_fold)
        accuracy.append(model.score(x_test_fold, y_test_fold))

    # Print the output.
    print(f'The model {model_name} has an Average Accuracy:', round(mean(accuracy) * 100, 2), '%')
    print('\nMaximum Accuracy that can be obtained:', round(max(accuracy) * 100, 2), '%')
    print('\nStandard Deviation:', stdev(accuracy))
    print('\n\n\n')

    ratio=round(max(accuracy) * 100, 2)

    detection_results_model.objects.create(names=model_name.replace("'",""), ratio=ratio)

def train_model(request):
    obj=''
    models = {}
    models["'Logistic Regression'"] = LogisticRegression(random_state=12345)
    models["'K Nearest Neighbour'"] = KNeighborsClassifier()
    models["'Decision Tree'"] = DecisionTreeClassifier(random_state=12345)
    models["'Random Forest'"] = RandomForestClassifier(random_state=12345)
    models["'SVM'"] = SVC(gamma='auto', random_state=12345)
    models["'XGB'"] = GradientBoostingClassifier(random_state=12345)

    for key, values in models.items():
        train_model1(key, values)

    obj = detection_results_model.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})














