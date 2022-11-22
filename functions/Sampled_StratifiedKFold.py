# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection    import train_test_split, StratifiedKFold
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import accuracy_score, confusion_matrix

from copy import deepcopy

from imblearn.over_sampling     import RandomOverSampler
from imblearn.under_sampling    import RandomUnderSampler
from imblearn.pipeline          import Pipeline

# %% 
def SSKFold(X, y
                    , model              = RandomForestClassifier()
                    , number_folds       = 5
                    , shuffle            = True
                    , scoring_criteria   = accuracy_score
                    , undersample_method = RandomUnderSampler(sampling_strategy=.8)
                    , oversample_method  = RandomOverSampler(sampling_strategy='minority')
                    , random_seed        = 1
                    , hold_out_size      = 0.3
                    , cf                 = False):

    # Train/Test Split of Dataframes & 
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size = hold_out_size, random_state = random_seed)

    # Reset Indices of provided arrays
    X_train.reset_index(drop = True, inplace = True)
    y_train.reset_index(drop = True, inplace = True)

    ################
    # DEFINE STRATIFICATION & PREPARE PIPELINE
    ################
    kf = StratifiedKFold(n_splits=number_folds, random_state = random_seed, shuffle = shuffle)    
    pipeline = Pipeline(steps=[('u', undersample_method), ('o', oversample_method)])

    ################
    # LISTS
    ################
    score_list  = []
    model_list  = []

    ################
    # STRATIFICATION AND MODEL FITTING
    ################  
    for train_fold_index, val_fold_index in kf.split(X_train, y_train):
        
        # Set the training data for the fold, and the validation data for the fold
        X_train_fold, y_train_fold  = X_train.iloc[train_fold_index], y_train[train_fold_index]
        X_val_fold  , y_val_fold    = X_train.iloc[val_fold_index],   y_train[val_fold_index]

        # Sythnetic Sampling of only the Training Data within the fold
        X_train_fold_sampled, y_train_fold_sampled = pipeline.fit_resample(X_train_fold, y_train_fold)

        # Fit the model, and append a deepcopy to the model list
        model_obj = model.fit(X_train_fold_sampled, y_train_fold_sampled)
        model_list.append(deepcopy(model_obj))

        # Score the model on the (non-upsampled) validation data, and append the score to the list
        score = scoring_criteria(y_val_fold, model_obj.predict(X_val_fold))
        score_list.append(score)

    ################
    # GET AVERAGE SCORE WEIGHTED PROBABILITY FOR EACH ROW
    ################

    # Define Empty list of probabilities
    y_probs = pd.DataFrame()
    y_proba = pd.DataFrame()

    # For each model is the model_list, predict probabilities for the hold out
    for i in range(0, len(model_list)):
        y_probs[i] = model_list[i].predict_proba(X_holdout)[:, 1]
    
    # Average Probabilities for each record
    y_proba['y_pred_wa'] = (y_probs * score_list).sum(axis = 1) / sum(score_list)
    y_proba['y_actual'] = y_holdout.reset_index(drop=True)

    ################
    # DETERMINE TUNING THRESHOLD 
    ################ 

    # Create list of thresholds to test, and create dataframe of empty scores (for each threshold)
    thresholds = np.arange(0.0, 1.0, 0.001)
    scores = np.zeros(shape=(len(thresholds)))

    # For each row in thresholds, 
    for index, elem in enumerate(thresholds):
        
        # Determine if the probability is above or below threshold (turn into binary)
        y_pred_prob = (y_proba['y_pred_wa'] > elem).astype('int')
        
        # Determine score after correction
        scores[index] = scoring_criteria(y_proba['y_actual'], y_pred_prob)
    
    ################
    # FIND THE BEST THRESHOLD
    ################ 
    pd.set_option('display.max_rows', None)

    # Find the optimial threshold (threshold that provides the max score)
    index = np.argmax(scores)
    thresholdOpt = round(thresholds[index], ndigits = 5)
    scoreOpt = round(scores[index], ndigits = 5)

    # Add boolean field for y_probs
    y_proba['y_pred'] = (y_proba['y_pred_wa'] >= thresholdOpt).astype(int)


    if cf == True:
        # Create Confusion Matrix Object
        cf_matrix = confusion_matrix(y_proba['y_actual'], y_proba['y_pred'])
        
        # Create Figure from Object
        ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
        ax.set_title('Confusion Matrix\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')
        ax.xaxis.set_ticklabels(['False','True'])
        ax.yaxis.set_ticklabels(['False','True'])

        # Show
        plt.show()

    return thresholdOpt, scoreOpt, score_list, model_list, y_proba, y_probs
