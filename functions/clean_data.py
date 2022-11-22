# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# %%
def clean_df(df_all):
    # Filling the missing values in Age with the medians of Sex and Pclass groups
    df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    df_all['isMinor'] = np.where(df_all['Age'] <= 16, 1, 0)

    # Filling the missing values in Embarked with S
    df_all['Embarked'] = df_all['Embarked'].fillna('S')

    # Filling the missing value in Fare with the median Fare of 3rd class alone passenger
    med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    df_all['Fare'] = df_all['Fare'].fillna(med_fare)

    # Creating Deck column from the first letter of the Cabin column (M stands for Missing)
    df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    
    # Passenger in the T deck is changed to A
    idx = df_all[df_all['Deck'] == 'T'].index
    df_all.loc[idx, 'Deck'] = 'A'

    df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
    df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
    df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')

    df_all.drop(['Cabin'], inplace=True, axis=1)

    # FEATURE ENGINEERING
    df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
    df_all['Age'] = pd.qcut(df_all['Age'], 10)

    df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 9:'Large', 10: 'Large', 11: 'Large'}
    df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)

    df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

    df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    df_all['Is_Married'] = 0
    df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1

    df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
    df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

    non_numeric_features = ['Age', 'Fare']

    for feature in non_numeric_features:        
        df_all[feature] = LabelEncoder().fit_transform(df_all[feature])


    drop_cols = ['Family_Size', 'Ticket', 'Name']

    df_all.drop(columns=drop_cols, inplace=True)
# %%
