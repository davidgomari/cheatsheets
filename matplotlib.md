# Mathematic Plot Library

```python
import matplotlib.pyplot as plt
```


## 1 General 

| **Description** | **Code** |
|:---------------:|:--------:|
| Set The Title | `plt.title("Title")` |
| Set the x-axis label | `plt.xlabel('Label')` |
| Set the y-axis label | `plt.ylabel('Label')` |
| Show the Legend | `plt.legend()` |
| Show the plt | `plt.show()` |

### **kwargs
1. `label` is a string that will be displayed in the legend.


### Subplots

| **Method** | **Type** | **Default** | **Description** |
|:----------:|:--------:|:-----------:|:---------------:|
| nrows, ncols | int | 1 | Number of rows/columns of the subplot grid. |
| sharex, sharey | bool | False | When subplots have a shared x-axis along a column, only the x tick labels of the bottom subplot are created. Similarly, when subplots have a shared y-axis along a row, only the y tick labels of the first column subplot are created. |
| ax | ~ | ~ | ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created. to access each ax you can use index like `ax[0]` and plot on it like `ax[0].plot(x,y)`. |



```python
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12,3), sharex=False, sharey=True)
```


## 2 Plots

either one of `plt` and `axs` can be used.

### Scatter plot

```python
plt.scatter(x, y, marker, c, **kwargs)
```

| **Description** | Point | Pixel | Circle | Square | Plus | X | Star | vline | hline | Up | Down |
|:---------------:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Symbol** | ![alt text](image.png) | ![alt text](image-1.png)| ![alt text](image-2.png) | ![alt text](image-10.png) | ![alt text](image-3.png) | ![alt text](image-4.png) | ![alt text](image-9.png) | ![alt text](image-5.png) | ![alt text](image-6.png) | ![alt text](image-7.png) | ![alt text](image-8.png) |
| **Marker** | "." | "," | "o" | "s" | "+" | "x" | "*" | "\|" | "_" | "^" | "v" |

`c` is color it can be `r` or `b` and...



