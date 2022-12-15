import pandas as pd
import numpy as np
import math

"""

Input variables:

bank client data:
- age (numeric)
- job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
- default: has credit in default? (categorical: 'no','yes','unknown')
- housing: has housing loan? (categorical: 'no','yes','unknown')
- loan: has personal loan? (categorical: 'no','yes','unknown')

related with the last contact of the current campaign:
- contact: contact communication type (categorical: 'cellular','telephone')
- month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
- duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

other attributes:
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- previous: number of contacts performed before this campaign and for this client (numeric)
- poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

social and economic context attributes
- emp.var.rate: employment variation rate - quarterly indicator (numeric)
- cons.price.idx: consumer price index - monthly indicator (numeric)
- cons.conf.idx: consumer confidence index - monthly indicator (numeric)
- euribor3m: euribor 3 month rate - daily indicator (numeric)
- nr.employed: number of employees - quarterly indicator (numeric)


Output variable (desired target):

- y - has the client subscribed a term deposit? (binary: 'yes','no')

Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#

"""


def get_bank_additional():
    
    # import data
    df = pd.read_csv('data/bank-additional-full.csv', delimiter=';')

    # replace unknown with nulls, and drop rows
    df.replace('unknown', np.nan, inplace=True)
    df.dropna(inplace=True)

    # CLEAN DATA ----

    ## Normalize age
    df['age'] = df['age'] / df['age'].max()

    ## Categorize Job
    job_categories = {
        "housemaid": "bluecollar",
        "services": "services",
        "admin.": "admin",
        "technician": "technician",
        "blue-collar": "bluecollar",
        "unemployed": "none",
        "retired": "none",
        "entrepreneur": "entrepreneur",
        "management": "management",
        "student": "none",
        "self-employed": "entrepreneur"
    }

    df['isStudent'] = np.where(df['job'] == 'student', 1, 0)
    df['isEmployed'] = np.where((df['job'] == 'student') | (df['job'] == 'retired') | (df['job'] == 'unemployed'), 0, 1)
    df['job'].replace(job_categories, inplace=True)

    df = pd.get_dummies(df, prefix='job', columns=['job'])

    ## Create Sin/Cos of Month/Day
    month_dict = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, 
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

    df['month'].replace(month_dict, inplace=True)
    df["month_sin"] = df["month"].apply(lambda x: math.sin(math.radians(x * 30)))
    df["month_cos"] = df["month"].apply(lambda x: math.cos(math.radians(x * 30)))

    day_dict = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7} 

    df['day_of_week'].replace(day_dict, inplace=True)
    df["dow_sin"] = df["day_of_week"].apply(lambda x: math.sin(math.radians(x * 7)))
    df["dow_cos"] = df["day_of_week"].apply(lambda x: math.cos(math.radians(x * 7)))

    ## Hot Encode Marriage
    df['isDivorced'] = np.where(df['marital'] == 'divorced', 1, 0)
    df['isMarried'] = np.where(df['marital'] == 'married', 1, 0)

    ## Hot Encode Education
    edu_categories = {
        "illiterate": "none",
        "basic.4y": "elementary",
        "basic.6y": "elementary",
        "basic.9y": "elementary",
        "high.school": "high.school",
        "professional.course": "higher.education",
        "university.degree": "higher.education",
    }

    df['education'].replace(edu_categories, inplace=True)
    df['hasSchooling'] = np.where(df['education'] == 'none', 0, 1)
    df = pd.get_dummies(df, prefix='education', columns=['education'])

    ## Boolean-ize default, housing, loan, contact
    df.replace({'yes': 1, 'no': 0, 'telephone': 1, 'cellular': 0}, inplace=True)
    df.rename(columns={'default': 'isDefault', 'housing': 'hasMortgage', 'loan': 'hasLoan', 'contact': 'isTelephone'}, inplace=True)

    # Hot Encode poutcome
    df['priorCampaign'] = np.where(df['poutcome'] == 'nonexistent	', 0, 1)
    df['priorSuccess'] = np.where(df['poutcome'] == 'success', 1, 0)

    ## Standardize social/economic variables
    df["emp.var.rate"] = (df["emp.var.rate"] - df["emp.var.rate"].mean()) / df["emp.var.rate"].std()
    df["cons.price.idx"] = (df["cons.price.idx"] - df["cons.price.idx"].mean()) / df["cons.price.idx"].std()
    df["cons.conf.idx"] = (df["cons.conf.idx"] - df["cons.conf.idx"].mean()) / df["cons.conf.idx"].std()
    df["euribor3m"] = (df["euribor3m"] - df["euribor3m"].mean()) / df["euribor3m"].std()
    df["numEmployed"] = (df["nr.employed"] - df["nr.employed"].mean()) / df["nr.employed"].std()

    ## Drop, Rename, and Order Columns
    df.drop(columns=['marital', 'nr.employed', 'poutcome', 'job_none', 'education_none', 'duration'], inplace=True)
    df.rename(columns={'previous': 'prevContacts', 'pdays': 'lastContact', 'y': 'purchased'}, inplace=True)

    cols = [
        'age'
        , 'isDivorced'
        , 'isMarried'
        , 'hasSchooling'
        , 'education_elementary'
        , 'education_high.school'
        , 'education_higher.education'
        , 'isStudent'
        , 'isEmployed'
        , 'job_admin'
        , 'job_bluecollar'
        , 'job_entrepreneur'
        , 'job_management'
        , 'job_services'
        , 'job_technician'
        , 'isDefault'
        , 'hasMortgage'
        , 'hasLoan'
        , 'isTelephone'
        , 'month'
        , 'month_sin'
        , 'month_cos'
        , 'day_of_week'
        , 'dow_sin'
        , 'dow_cos'
        , 'campaign'
        , 'priorCampaign'
        , 'priorSuccess'
        , 'prevContacts'
        , 'lastContact'
        , 'emp.var.rate'
        , 'cons.price.idx'
        , 'cons.conf.idx'
        , 'numEmployed'
        , 'euribor3m'
        , 'purchased'
    ]

    df = df[cols]

    # Get X/y Datasets
    X = df.drop(columns=['purchased'])
    y = df['purchased']

    return X, y, df

