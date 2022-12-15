import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

"""
Context
Although this dataset was originally contributed to the UCI Machine Learning repository nearly 30 years ago, 
mushroom hunting (otherwise known as "shrooming") is enjoying new peaks in popularity. Learn which features 
spell certain death and which are most palatable in this dataset of mushroom characteristics. 
And how certain can your model be?

Content
This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms 
in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American 
Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown 
edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly 
states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets 
three, let it be'' for Poisonous Oak and Ivy.

Source: https://www.kaggle.com/datasets/uciml/mushroom-classification

"""

def find_pca_components(X):
    # Center and scale data
    X_pca = X[X.columns[1:]]
    X_pca = X_pca.apply(lambda x: (x - x.mean()) / x.std())

    # Calculate eigenvalues and eigenvectors of the covariance matrix
    cov_matrix = np.array(X_pca.cov())
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Check that the sum of the eigenvalues equals the sum of the variances
    assert np.isclose(eigenvalues.sum(), cov_matrix.diagonal().sum())

    # Calculate the proportion of variance explained by each eigenvector
    ve = eigenvalues / eigenvalues.sum()

    # Plot the proportion of variance explained by each eigenvector
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    # Scree plot
    plt.subplot(1, 2, 1)
    plt.plot(ve)
    plt.xlabel("Principal Component")
    plt.ylabel("Proportion of Variance Explained")
    plt.ylim(0, 1)
    plt.title("Scree plot")

    # Cumulative proportion of variance explained
    plt.subplot(1, 2, 2)
    plt.plot(ve.cumsum())
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Variance Explained")
    plt.ylim(0, 1)
    plt.title("Scree plt")

    plt.show()

def get_mushrooms(pca_components, minority_frac):
    # import data
    mush = pd.read_csv('data/mushrooms.csv')

    # create X/y dataframes
    X = pd.DataFrame(mush.drop('class', axis=1))
    y = pd.DataFrame(mush['class'])

    # Label encode X's
    Encoder_X = LabelEncoder()
    for col in X.columns:
        X[col] = Encoder_X.fit_transform(X[col])

    # Recode X, and rename column
    y['class'] = np.where(y['class'] == 'p', 1, 0)
    y.rename(columns={'class': 'isPoison'}, inplace=True)

    # Get dummies for X
    X = pd.get_dummies(X, columns=X.columns, drop_first=True)

    # Scale X
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Print scree plots
    # find_pca_components(X)

    # Reduce Dimensionality
    # --> reducing from 95 to 40
    pca = PCA(n_components = pca_components)
    X = pca.fit_transform(X)

    # Convert X to dataframe and rename columns
    X = pd.DataFrame(X)
    X.rename(columns=lambda x: 'pca_{}'.format(x+1), inplace=True)

    # Create df by joining y, and X
    df = y.join(X)

    # Randomly sample 75% of isPoison to make it more imbalanced
    dfx = df.sample(frac = minority_frac, replace=False, random_state=1)
    df = df.drop(dfx[dfx["isPoison"] == 1].index)

    # Recreated X, y
    X = df.drop(columns=['isPoison'])
    y = df['isPoison']

    return X, y, df