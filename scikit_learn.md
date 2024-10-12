# Scikit-learn

[Scikit-learn Documentation](https://scikit-learn.org/stable/index.html)

All the models have the following structure and just the `PCK` and `MODEL` will change.

```python
from sklearn.PCK import MODEL
model = MODEL(max_iter=1000)
model.fit(X_train, y_train)
y_prediction = model.predict(X) 
```


| Method | Description |
|:-:|:-:|
| `model.fit(X, y)` | train the model |
| `model.predict(X)` | make a prediction |
| `model.score(X,y)` | get model accuracy on these new input data |
| `model.n_iter_` | Number of iterations completed <= max_iter |
| `model.t_` | Number of weights updated |
| `model.intercept_` | `b` final value |
| `model.coef_` | `W` final value |

## 1 Models

### 1.1 Linear Models

```python
from sklearn.linear_model import MODEL
```

| Model Name | Sklearn Name (MODEL) | Description |
|:-:|:-:|:-:|
| Linear Regression | `LinearRegression` | Simple Linear Regression |
| Stochastic Gradient Descent Regressor | `SGDRegressor` | Linear Regression with Regularization |
| Logistic Regression | `LogisticRegression` | Classification with Regularization |

### 1.2 Tree Based

| Name | Library | Model Name | Hyperparameters |
|:-:|:-:|:-:|:-:|
| Decision Tree | `sklearn.tree` | `DecisionTreeClassifier` | `min_samples_split`: The minimum number of samples required to split an internal node. `max_depth`: The maximum depth of the tree. `min_impurity_decrease`: (default=0.0) A node will be split if this split induces a decrease of the impurity greater than or equal to this value. `max_features`: The number of features to consider when looking for the best split. Options: an integer ,'sqrt', 'log2', and 'None' for all features. |
| Random Forest | `sklearn.ensemble` | `RandomForestClassifier` | One additional hyperparameter: `n_estimators` (default = 100) which is the number of Decision Trees that make up the Random Forest. **Parallel training:** setting `n_jobs` higher will increase how many CPU cores it will use. |


### 1.3 Not Sklearn models

It can be used just like sklearn models.

```python
from LIBRARY import MODEL_NAME
```

| Name | Library | Model Name | Parameters |
|:-:|:-:|:-:|:-:|
| Regression XGBoost | `xgboost` | `XGBRegressor` | `n_estimators`, `learning_rate` |
| Classification XGBoost | `xgboost` | `XGBClassifier` | same |




## 2 Preprocessing
- **Normalization**

    1)    **Normalization:** `Normalizer` scales numeric features to [0,1]. Normalizer scales samples (rows) individually to unit norm, focusing on the magnitude of samples rather than features. It scales each data sample (row) to have a unit norm (typically L2 norm).

    2)   **Standardization (Z-score Normalization):** `StandardScaler` mean = 0, std = 1.

    3)   **MinMaxScaler:** `MinMaxScaler` It is useful when you want all features to lie within a certain range, especially when the distribution of your features may differ significantly. $$ X_{scaled} = {{X − X_{min}} \over {X_{max} − X_{min}}} $$


    If the model is trained with normalized input features, you should normalize the input (x) before feeding it to the model for prediction.

    ```python
    from sklearn.preprocessing import StandardScaler
    norm = StandardScaler()
    X_norm = norm.fit_transform(X_train)
    ```

-  **Polynomial Features**
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    # Instantiate the class to make polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False) # (x, x^2)
    # Compute the features and transform the training set
    X_train_mapped = poly.fit_transform(x_train) 
    ```


## 3 Model Selection

- **Train Test Split**
    ```python
    from sklearn.model_selection import train_test_split
    # random state: Pass an int for reproducible output across multiple function calls
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.4, random_state=1)
    ```

- **GridSearchCV (Hyperparameter Tunning)**

    Ideally, we would want to check every combination of values for every hyperparameter that we are tuning and choose the best one.
    
    Change `MODEL` to your selected model. `parameters` is a Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    
    `scoring` is the Strategy to evaluate the performance of the cross-validated model on the test set. If scoring represents multiple scores, use a list or tuple of unique strings ([click](https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring) to see all the metrics available).
    
    `refit` (default: `True`) Refit an estimator using the best found parameters on the whole dataset. For multiple metric evaluation, this needs to be a str denoting the scorer that would be used to find the best parameters for refitting the estimator at the end. 
    
    `cv` determines the k-Fold Cross-Validation splitting strategy parameter `k`. Possible inputs for cv are: `None`, to use the default 5-fold cross validation, and an integer, to specify the number of folds. `n_jobs=-1` will allow to use all the CPU cores (be carefull).
    
    for more details [click here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

    ```python
    from sklearn.model_selection import GridSearchCV

    GS = GridSearchCV(
        estimator=MODEL(),
        param_grid=parameters,
        scoring='accuracy',
        refit='accuracy',
        cv=4,
        verbose=4,
        n_jobs=2
    )

    GS.fit(X_train, y_train)
    ```

    `GS` is fitted with best parameters and can be used for prediction.

## 4 Metrics

```python
from sklearn.metrics import METRIC_NAME
```
`m` is number of data in dataset.

- **Mean Squared Error** `mean_squared_error`

    This method can be used to compute MSE on some dataset. 
    $$J(\vec{w}, b) = \frac{1}{m}\left[\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2\right]$$

    ```python
    # this is and exmaple of computing MSE on Cross Validation dataset
    y_hat = model.predict(x_cv)
    J_mse_cv = mean_squared_error(y_cv, y_hat)
    ```

- **Accuracy Score** `accuracy_score`

    `accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)`
    
    In multilabel classification, this function computes subset accuracy. The best performance is 1 with `normalize == True` and the number of samples with `normalize == False`.

    $$accuracy = \frac{m - missClassifiedSamples}{m}$$


