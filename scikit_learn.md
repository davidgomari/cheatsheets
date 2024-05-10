# Scikit-learn

[Scikit-learn Documentation](https://scikit-learn.org/stable/index.html)

## 1 Models

<!-- - **Abbreviations** -->
| Abb | Description |
|:-:|:-:|
| SGD | Stochastic Gradient Descent |

<!-- - **Models Description**
    | Model | Description |
    |:-:|:-:|
    | Regression | | -->

### 1.1 Regression

- **SGD Regressor**

    | Method | Description |
    |:-:|:-:|
    | `sgdr.n_iter_` | Number of iterations completed <= max_iter |
    | `sgdr.t_` | Number of weights updated |
    | `sgdr.intercept_` | `b` final value |
    | `sgdr.coef_` | `W` final value |


    ```python
    from sklearn.linear_model import SGDRegressor
    sgdr = SGDRegressor(max_iter=1000)
    sgdr.fit(X_norm, y_norm)
    y_pred_sgd = sgdr.predict(X_norm)
    ```


## 2 Tools


### 2.1 Normalization

- **Z-score**

    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train)
    ```
