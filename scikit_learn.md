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

<!-- 
### 1.1 Regression
- **Stochastic Gradient Descent Regressor**

    ```python
    from sklearn.linear_model import SGDRegressor
    ```

### 1.2 Classification
- **Logistic Regression**

    ```python
    from sklearn.linear_model import LogisticRegression
    ``` -->


## 2 Preprocessing
- **Normalization**

    **Normalization:** `Normalizer` scales numeric features to [0,1].

    **Standardization (Z-score Normalization):** `StandardScaler` mean = 0, std = 1.

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

## 4 Metrics

- **Mean Squared Error**

    This method can be used to compute MSE on some dataset. `m` is number of data in dataset.
    $$J(\vec{w}, b) = \frac{1}{m}\left[\sum_{i=1}^{m}(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2\right]$$

    ```python
    from sklearn.metrics import mean_squared_error
    # this is and exmaple of computing MSE on Cross Validation dataset
    y_hat = model.predict(x_cv)
    J_mse_cv = mean_squared_error(y_cv, y_hat)
    ```
