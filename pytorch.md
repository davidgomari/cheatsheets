# PyTorch

```python
import torch
print(torch.__version__)
```

## Tensor

- a multidimensional grid of values, all of the same type.
- `a` is a tensor.
- **`STORAGE` tag**: output tensor shares the same data as the input. so changes to one will affect the other and vice-versa


### 1 Creating

| Description | Code/Example |
|:-:|:-:|
| creating a 1-dimentional tensor | `a = torch.tensor([1, 2, 3])` |
| creating a 2-dimentional tensor | `b = torch.tensor([[1, 2, 3], [4, 5, 5]])` |
| Zero Tensor | `torch.zeros(shape)` |
| create new zero tensor with the same shape and type as a given tensor `a` | `torch.zeros_like(a)` |
| Zero tensor with same `dtype` but not necessary same shape | `a.new_zeros(new_shape)` |
| One Tensor | `torch.ones(shape)` |
| Random Tensor | `torch.rand(shape)` |
| Identity Matrix | `torch.eye(int)` |
| Empty Tensor | `torch.empty(shape)` |


### 2 Arguments

`torch.tensor(array, dtype, device)`

| Name | Description | Values |
|:-:|:-:|:-:|
| `dtype` | is used to explicitly specify a datatype. | `torch.float32`, `torch.float16`, `torch.int64`, `torch.int32`, `torch.uint8`, `torch.bool` |
| `device` | Specifies the device where the tensor is stored. A tensor on a CUDA device will automatically use that device to accelerate all of its operations. | `cuda`, `cpu` |

### 3 Attributes

| Name | Description | Code/Example |
|:-:|:-:|:-:|
| **Rank** | Number of dimensions | `a.dim()` |
| **Shape** | A tuple of integers giving the size of the array along each dimension | `a.shape` |
| **Item** | Acessing an element from a PyTorch Tensor | `a[0].item()` or `b[1,2].item()` |
| **dtype** | returns data type of tensor | `a.dtype` | 
| **device** | returns in what device the tensor is stored | `a.device` |

### 4 Methods

| Name | Description | Code/Example |
|:-:|:-:|:-:|
| **To** | cast a tensor to another **datatype** | `a.to(dtype)` |
| **To** | change the **device** of a tensor. | `a.to(device)` | 
| **Clone** | make a copy of a tensor (with different pointer) | `a.clone()` |
| **View `STORAGE`**  | Reshape tensor `a` and returns a new tensor. **Flatten(n,):** `a.view(-1)`. **Row Vector(1,n):** `a.view(1,-1)`. | `a.view(new_shape)` |
| **Transpose `STORAGE`** | return transposed tensor | `a.t()` |
| **Move new tensor** | These methods create a new clone tensor in another device based on tensor `a`. | `a.cuda()` `a.cpu()` |
| **Repeat** | Repeats this tensor along the specified dimensions. | `a.repeat(sizes*)` |
| **Chunk `STORAGE`** | Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor. `chunks` is number of chunks to return. `dim` is dimension along which to split the tensor. for splitting x_train use `dim=0`. | `a.chunk(chunks, dim)` |
| **Cat** | Concatenates the given sequence of seq tensors in the given dimension. `tensors` (sequence of Tensors) any python sequence of tensors of the same type. `dim` (int, optional) the dimension over which the tensors are concatenated. | `a.cat(tensors, dim=0)` |


- **Example:** this code repeat vector `v` in 4 rows and in each row, 2 times.

  ```python
  v = torch.tensor([1,2])
  v.repeat(4,2)
  # tensor([[ 1,  2,  1,  2],
  #         [ 1,  2,  1,  2],
  #         [ 1,  2,  1,  2],
  #         [ 1,  2,  1,  2]])
  ```



### 5 Slicing `STORAGE`

Slicing a tensor returns a view into the same data, so modifying it will also modify the original tensor. 

```python
# Two ways of accessing a single row:
row_r1 = a[1, :]    # Rank 1 view of the second row of a     shape=(n,)
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a     shape=(1,n)

# Two ways of accessing a single column:
col_r1 = a[:, 1]    # shape=(n,)
col_r2 = a[:, 1:2]  # shape=(n,1)
```

### 6 Tensor indexing

#### Integer tensor indexing
    
-    We can use index arrays to index tensors; this lets us construct new tensors with a lot more flexibility than using slices.

-    Given index arrays `idx0` and `idx1` with `N` elements each, `a[idx0, idx1]` is equivalent to:

        ```python
        torch.tensor([
        a[idx0[0], idx1[0]],
        a[idx0[1], idx1[1]],
        ...,
        a[idx0[N - 1], idx1[N - 1]]
        ])
        ```

-    Example

        ```python
        # reverse rows of x and save in y
        y = torch.zeros_like(x)
        M, N = x.shape
        idx = [i for i in range(M-1,-1,-1)]
        y = x[idx]
        ```

#### Boolean tensor indexing

-   Lets you pick out arbitrary elements of a tensor according to a boolean mask (satisfying some condition).
-   Example

    ```python
    # elements <= 3 set to zero:
    a[a <= 3] = 0

    # Return the sum of all the positive values in the input tensor x.
    pos_sum = sum(x[x > 0]).item()
    ```



### 7 Operations

- `x` is a tensor.

#### 7.1 Mathematics Operations

- You can find a full list of all available mathematical functions in [the documentation](https://pytorch.org/docs/stable/torch.html#pointwise-ops);

| Type | Name  | Code |
|:-:|:-:|:-:|
| Math | Sqrt | `x.sqrt()` or `torch.sqrt(x)` |
| Math | Sine | `x.sin()` or `torch.sin(x)` |
| Math | Cosine | `x.cos()` or `torch.cos(x)` |

#### 7.2 Reduction Operations

- We may sometimes want to perform operations that aggregate over part or all of a tensor, such as a summation; these are called **reduction** operations.
- We can use the `.sum()`, `.mean()`, `.max()`, and other methods (or eqivalently `torch.sum`, and etc.) to reduce either an entire tensor, or to reduce along only one dimension of the tensor using the `dim` argument
- To understand reduction, think about the shapes of the tensors involved. *Example:* After summing with `dim=d`, the dimension at index `d` of the input is **eliminated** from the shape of the output tensor.
- Reduction operations reduce the rank of tensors: the dimension over which you perform the reduction will be removed from the shape of the output. If you pass `keepdim=True` to a reduction operation, the specified dimension will not be removed; the output tensor will instead have a shape of 1 in that dimension.
- Find more reduction operations in [the documentation](https://pytorch.org/docs/stable/torch.html#reduction-ops).

| Type | Name | Desc | Code |
|:-:|:-:|:-:|:-:|
| Reduction | **Sum** | | `x.sum()` or `torch.sum(x)` |
| Reduction | **Mean** | | `x.mean()` or `torch.mean(x)` |
| Reduction | **Max** | | `x.max()` or `torch.max(x)` |
| Reduction | **Min** | also can do both `.amin()` and `.argmin()` if argument `dim` is provided. | `x.min()` or `torch.min(x)` |
| Reduction | **ArgMin** | Returns the indices of the minimum value(s) of the flattened tensor or along a dimension | `x.argmin()` or `torch.argmin(x)` |
| Reduction | **aMin** | Returns the minimum value of each slice of the `input` tensor in the given dimension(s) `dim`. | `x.amin()` or `torch.amin(x)` |
| Reduction | **TopK** | Returns the `k` largest elements of the given input tensor along a given dimension. If `dim` is not given, the last dimension of the input is chosen. If `largest` is `False` then the k smallest elements are returned. | `(values, indices) = x.topk(k, dim=None, largest=True)` |
| Reduction | **Binary Counting** | Count the frequency of each value in an array of non-negative ints. | `x.bincount()` |

- Example:

```python
# Return a copy of the input tensor x, where the minimum value along each row has been set to 0.
idx_min = x.argmin(dim=1)
y = x.clone()
M, _ = x.shape
y[[i for i in range(M)], idx_min] = 0
```

#### 7.3 Matrix Operations

The most commonly used are:

- `.dot`: Computes inner product of vectors
- `.mm`: Computes matrix-matrix products (multiplication)
- `.mv`: Computes matrix-vector products (Itself matches the shapes of vector and matrix)
- `.addmm` / `.addmv`: Computes matrix-matrix and matrix-vector multiplications plus a bias
- `.bmm` / `.baddmm`: Batched versions of `.mm` and `.addmm`, respectively
- `.matmul`: General matrix product that performs different operations depending on the rank of the inputs. Confusingly, this is similar to `np.dot` in numpy.
- Find a full list of the available linear algebra operators in [the documentation](https://pytorch.org/docs/stable/torch.html#blas-and-lapack-operations).

### 8 Broadcasting

-    **Example 1:** Multiply a tensor by a set of constants
        ```python
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # x has shape (2, 3)
        c = torch.tensor([1, 10, 11, 100])        # c has shape (4)
        # Reshape c from (4,) to (4, 1, 1)
        # The result of the broadcast multiplication between tensor of shape
        #    (4, 1, 1) and (1, 2, 3) has shape (4, 2, 3)
        y = c.view(-1, 1, 1) * x
        ```


## Running on GPU

We can check whether PyTorch is configured to use GPUs:
```python
if torch.cuda.is_available():
  print('PyTorch can use GPUs!')
else:
  print('PyTorch cannot use GPUs.')
```






