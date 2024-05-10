# NumPy

```python
import numpy as np
```

[NumPy Documentation](https://numpy.org/doc/stable/)

## 1 Creation

Data creation routines in NumPy will generally have a first parameter which is the shape of the object. This can either be a single value for a 1-D result or a tuple (n,m,...) specifying the shape of the result. Below are examples of creating vectors using these routines.

tips:
- `!` means the shape can only be `(n,)`.

| Description | Code |
|:-:|:-:|
| Create array from python list  | `np.array(list, dtype)` |
| Zeros Vector | `np.zeros(shape, dtype)` | 
| Ones Vector | `np.ones(shape, dtype)` |
| Random Vector (`0 < val < 1`) | `np.random.random_sample(shape, dtype)` |
| `!` Create evenly spaced values | `np.arange(start, stop, step, dtype)` |

## 2 Operations

| Description | Code |
|:-:|:-:|
| Calculate Sum | `np.sum()` |
| Calculate Mean | `np.mean()` |
| Vector Dot Product | `np.dot(a,b)` or `a @ b`|
| Reshape | `ndarray.reshape(shape)` |

**Examples:**

```python
# a vector of shape (9,) -> a matrix of shape (3,3)
vector = np.arange(9).reshape((3,3))
# automaticaly compute #rows given vector size and #cols
vector = np.arange(9).reshape(-1,3)
```


## 3 Usefull

1. **Slicing:** slicing creates an array of indices using a set of three values (`start:stop:step`). A subset of values is also valid. Its use is best explained by example:
    ```python
    # vector 2-D slicing operations

    a = np.arange(20).reshape(-1, 10)
    # a = [ 0  1  2  3  4  5  6  7  8  9]
    #     [10 11 12 13 14 15 16 17 18 19]

    b = a[:, 2:7:2]
    # b = [ 2  4  6]
    #     [12 14 16]
    ```

2. **Feature Engineering:** Create Input feature (X) by combining or transforming original features:

    `np.c_()`: Stack 1-D arrays as columns into a 2-D array.
    ```python
    # suppose `x` is one of our features with shape (m,)
    # `m` is number of training examples
    # `X` will be a matrix with shape (m,4)

    X = np.c_[x, x**2, x**3]
    ```
