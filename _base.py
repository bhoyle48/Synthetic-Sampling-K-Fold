# %%
import numpy as np
import pandas as pd

from sklearn.model_selection    import train_test_split, StratifiedKFold
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import accuracy_score, balanced_accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score, confusion_matrix

from copy import deepcopy

from imblearn.over_sampling     import RandomOverSampler
from imblearn.under_sampling    import RandomUnderSampler
from imblearn.pipeline          import Pipeline



class SyntheticStratifiedKFold:
    def __init__(
        self
        , model_type
        , scoring = f1_score
        , folds = 5
        , shuffle = True
        , undersampler = RandomUnderSampler(sampling_strategy=0.8)
        , oversampler  = RandomOverSampler(sampling_strategy='minority')
        , random_state = 1
    ):
        self.model_type = model_type
        self.scoring = scoring
        self.folds = folds
        self.shuffle = shuffle
        self.undersampler = undersampler
        self.oversampler  = oversampler
        self.random_state = random_state
        self.optimum_threshold = None
        self.models = []
        self.scores = []
        self.predictions = []
        self.probabilities = []

    def fit(self, X, y, holdout_size = 0.3):
        self.models = []
        self.scores = []      

        # Split
        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size = holdout_size, random_state = self.random_state) 
        
        # Reset Indice
        X_train.reset_index(drop = True, inplace = True)
        y_train.reset_index(drop = True, inplace = True)
        
        # Prepare Pipeline
        kf = StratifiedKFold(n_splits = self.folds, random_state = self.random_state, shuffle = self.shuffle)    
        pipeline = Pipeline(steps=[('u', self.undersampler), ('o', self.oversampler)])
        
        # Copy/Fit Model(s)
        for train_fold_index, val_fold_index in kf.split(X_train, y_train):
            
            # Set the training data for the fold, and the validation data for the fold
            X_train_fold, y_train_fold  = X_train.iloc[train_fold_index], y_train[train_fold_index]
            X_val_fold  , y_val_fold    = X_train.iloc[val_fold_index],   y_train[val_fold_index]

            # Sampling of Data within the fold
            X_train_fold_sampled, y_train_fold_sampled = pipeline.fit_resample(X_train_fold, y_train_fold)

            # Fit Model
            model_obj = self.model_type
            model_obj = model_obj.fit(X_train_fold_sampled, y_train_fold_sampled)

            # Copy and Append
            self.models.append(deepcopy(model_obj))
        
            # Score and Append
            self.scores.append(self.scoring(y_val_fold, model_obj.predict(X_val_fold)))

        # Predict using all models     
        y_proba = pd.DataFrame(data = self.predict_proba(X_holdout), columns = ['y_proba'])

        # List of Thresholds
        thresholds = np.arange(0.0, 1.0, 0.005)
        scores = np.zeros(shape=(len(thresholds)))

        # Optimize Threshold   
        for index, elem in enumerate(thresholds):
            y_proba['y_prob'] = np.where(y_proba['y_proba'] > elem, 1, 0)
            scores[index] = self.scoring(y_holdout, y_proba['y_prob'])

        # Find the optimial threshold (threshold that provides the max score)
        self.optimum_threshold = thresholds[np.argmax(scores)]


    def predict_proba(self, X):

        probs = pd.DataFrame()

        for i in range(0, len(self.models)):
            probs[i] = self.models[i].predict_proba(X)[:, 1]

        self.probabilities = (probs * self.scores).sum(axis = 1) / sum(self.scores)

        return self.probabilities


    def predict(self, X):

        # Get probabilities
        probabilities = self.predict_proba(X)

        # Get predictions
        self.predictions = (probabilities >= self.optimum_threshold).astype('int')

        return self.predictions