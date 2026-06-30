import os
import kagglehub
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import streamlit as st

path = "spscientist/students-performance-in-exams"
def file_finder():
    try:
        download = kagglehub.dataset_download(path)
        print('The Path is Download')
        print(f'Path:{download}')

        csv = None
        for file_suffix in os.listdir(download):
            if file_suffix.endswith(".csv"):
                csv = os.path.join(download,file_suffix)
                print(f'Found_CSV:{csv}')
                break
        if csv:
            d_s = pd.read_csv(csv)
            print(d_s.columns.tolist())
            return {
                'Data':d_s
            }
        else:
            print('No CSV is Not Found....')
            return None

    except Exception as e:
        print(f'The Error Occured:{e}')
        return None

def d_b_create(x):
    file = 'students.db'
    try:
        with sqlite3.connect(file) as conn:
            # return conn
            print('DB is connected')

            x.to_sql(
            name='student_performance',
            con = conn,
            if_exists= 'replace',
            index = False
            )
            print('The data is stored in Students_DB')
            query = "SELECT file FROM pragma_database_list WHERE name = 'main';"
            db_path = conn.execute(query).fetchone()[0]
            print(f'Path:{db_path}')

            count = conn.execute('Select Count(*) From student_performance').fetchone()[0]
            print(f'Total Count:{count}')
            return {
                'Connect':conn
            }

    except Exception as e:
        print(f'The error {e}')
def records(x):
    try:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width',200)
        pd.set_option('display.max_rows',50)
        df = pd.read_sql('Select * From student_performance',con=x)
        print('='*50)
        print('The Student DB Table:')
        print(df)
        return {
            'Data':df
        }
        # return df
    except Exception as e:
        print(f"Can't Execute the DB {e}")
def data_cleaning(x):
    print('Count Duplicate Values..:')
    dupli = x.duplicated().sum()
    print(dupli)
    print('Categorical and Numerical Column..:')
    cat_colum = [cat for cat in x.columns if x[cat].dtype == 'object' ]
    num_colum = [cat for cat in x.columns if x[cat].dtype != 'object' ]
    print(cat_colum)
    print(num_colum)
    print('Counting Unique Values...:')
    print(x[cat_colum].nunique())
    print(x[num_colum].nunique())
    print('Null Values..:')
    null_values = round((x.isnull().sum()/x.shape[0])*100,2)
    print(null_values)
    print('Data Types..:')
    print(x.dtypes)
    print('Basic Statistics...')
    print(x.describe())
    return {
        'Categorical':cat_colum,
        'Numerical':num_colum
    }
def EDA_chart(x,column_name):
    print('Outlier using InterQuartile Method...:')
    cat_column = column_name['Categorical']
    num_column = column_name['Numerical']
    for i in num_column:
        q1 = x[i].quantile(0.25)
        q2 = x[i].quantile(0.50)
        q3 = x[i].quantile(0.75)
        # print(q1,q2,q3)
        IQR = q3 - q1
        # print(IQR)
        lower = q1 - 1.5*IQR
        upper = q3 + 1.5*IQR
        # print(upper,lower)
        # outlier = df[(df['amount'] < lower) | (df['amount'] > higher)]
        outlier = x[(x[i]<lower)|(x[i]>upper)]
        if len(outlier) > 0:
            print(outlier)
        else:
            print('No Outlier Found...')
        print('Cleaned Data after using Outlier to find Far Data..:')
        df_cleaned = x[(x[i] >= lower) & (x[i] <= upper)]
        print(df_cleaned)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Boxplot — Outlier Detection', fontsize=14)

    for idx, i in enumerate(num_column):
        sns.boxplot(y=x[i], ax=axes[idx], color='skyblue')
        axes[idx].set_title(i)
        axes[idx].set_ylabel('Score')
    plt.tight_layout()
    # plt.show()
def feature_engineering(x,column_name):
    print('='*50)
    print('Feature Engineering')
    cat_column = column_name['Categorical']
    num_column = column_name['Numerical']
    # print(len(num_column))
    '''['math score', 'reading score', 'writing score']'''
    x['Average'] = (x['math score']+x['reading score']+x['writing score'])/len(num_column)
    print(x['Average'])
    x['Result'] = x['Average'].apply(lambda s:1 if s>=40 else 0)
    # print(x.columns)
    print(x['Result'].value_counts())
    def grade_column(y):
        if y >= 90:
            return 'A'
        elif y >=75:
            return 'B'
        elif y>=60:
            return 'C'
        elif y>=40:
            return 'D'
        else:
            return 'F'
    x['Grade']=x['Average'].apply(grade_column)
    # print(x['Grade'].value_counts())
    # print(x.head(5))
    print('='*50)
    print('Label Encoder')
    LE = LabelEncoder()
    for col in cat_column:
        x[col] = LE.fit_transform(x[col])
        print(col)
    x['Grade'] = LE.fit_transform(x['Grade'])
    print('✅ Encoded: Grade')
    print(x.head(3))
    return x
def train_model(x):
    print('=' * 50)
    print(x.columns)
    feature = ['gender', 'race/ethnicity',
               'parental level of education',
               'lunch', 'test preparation course']
    X = x[feature]
    y = x['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(len(X_train))
    print(len(X_test))

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    print(classification_report(
        y_test, y_pred, target_names=['Fail', 'Pass']
    ))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'],
                yticklabels=['Fail', 'Pass'])
    plt.title('📊 Confusion Matrix')
    plt.tight_layout()

    importance = pd.Series(
        model.feature_importances_, index=feature
    ).sort_values(ascending=False)
    print('\n🔍 Feature Importance:')
    print(importance)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importance.values, y=importance.index, palette='Blues_d')
    plt.title('🔍 Feature Importance')
    plt.tight_layout()

    with open('student_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('✅ Model saved as student_model.pkl!')
    return model
file_find = file_finder()
if file_find is not None:
    db = d_b_create(file_find['Data'])
    if db is not  None:
        db_record = records(db['Connect'])
        if db_record is not None:
            data_clean = data_cleaning(db_record['Data'])
            EDA = EDA_chart(db_record['Data'],data_clean)
            Fea_Eng = feature_engineering(x=db_record['Data'],column_name=data_clean)
            ml_record = train_model(db_record['Data'])
print('hello ')