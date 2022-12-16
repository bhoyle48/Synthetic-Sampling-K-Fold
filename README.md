## Synthetically Sampling K-Fold
This project spawned from needing a solution for handling imbalanced classes, while utilizing the K-Fold validation technique. 

The approach taken was implementing up-sampling of the minority class, in conjunction reducing the majority class. The problem however, was the introduction of creating duplicate records (via up-sampling) which then are split into both the training and testing test, making accurate evaluation impossible. 

Additionally, best practice for implementing synthetic sampling is to do so on the training set while leaving the true distribution for the testing set. This allows the users to evaluate how a model performs in a 'live' setting.

### Prepare Folds & Pipeline for Synthetic Sampling
```python
# Prepare Pipeline
kf = StratifiedKFold(n_splits = self.folds, random_state = self.random_state, shuffle = self.shuffle)
pipeline = Pipeline(steps=[('u', self.undersampler), ('o', self.oversampler)])
```
### Within each fold, complete synthetic sampling and fit model

```python
# Copy/Fit Model(s)
for train_fold_index, val_fold_index in kf.split(X_train, y_train):

	# Set the training data for the fold, and the validation data for the fold
	X_train_fold, y_train_fold = X_train.iloc[train_fold_index], y_train[train_fold_index]
	X_val_fold , y_val_fold = X_train.iloc[val_fold_index], y_train[val_fold_index]

	# Sampling of Data within the fold
	X_train_fold_sampled, y_train_fold_sampled = pipeline.fit_resample(X_train_fold, y_train_fold)

	# Fit Model
	model_obj = self.model_type
	model_obj = model_obj.fit(X_train_fold_sampled, y_train_fold_sampled)
```


### Maintain imbalance integrity of test set and score
```python
self.scores.append(self.scoring(y_val_fold, model_obj.predict(X_val_fold)))
```

### Utilize all models to predict test set
```python
self.predict_proba(X_holdout)
```

### Find optimal threshold for specific metric
```python
# List of Thresholds
thresholds = np.arange(0.0, 1.0, 0.005)
scores = np.zeros(shape=(len(thresholds)))

# Optimize Threshold
for index, elem in enumerate(thresholds):
y_proba['y_prob'] = np.where(y_proba['y_proba'] > elem, 1, 0)
scores[index] = self.scoring(y_holdout, y_proba['y_prob'])

# Find the optimial threshold (threshold that provides the max score)
self.optimum_threshold = thresholds[np.argmax(scores)]
```

