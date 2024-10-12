# pandas_cheatsheet
my personal pandas cheatsheet. first of all: `import pandas as pd`


## General Commands
| Usage | Command | Description |
|:-----:|:-------:|:-----------:|
| Loading | `df = pd.read_csv(filepath_or_buffer='path', sep=';')` | `sep` default is `','`. `filepath_or_buffer` can be a download link to a CSV file |
| Save as CSV | `df.to_csv('file.csv')` | convert the DF to a CSV file and save it in current directory |
| Save as Excel | `df.to_excel(path, index=True)` | Save DF as EXCEL file. |
| Shape | `df.shape` | return a tuple (`#rows`, `#columns`) |
| Information | `df.info()` | All information about our CSV file we've loaded in. `object` data type is usually a string. |
| Description | `df.describe()` | Get a statistical description about all columns like min, max, std, avg, and more. |
| Display Rows | `df` or `df.head()` `df.tail()` or `df.head(10)` | changing the maximum displayed #columns (20 is default) to an arbitrary integer `a`: `pd.set_option('display.max_columns', a)` and for #rows: `pd.set_option('display.max_rows', a)` |
| Display in Python IDE | `display(df)` | We can display a DataFrame in the form of a table with borders around rows and columns. first, insert this code: `from IPython.display import display` |
| Plot | `df.plot()` | plots all the DF columns. better, first of all, select columns we care about and then use the `plot` method. |
| Index | `df.index` | the CK for identifying the rows |


## DataFrame
A data frame is like a Python dictionary. each `key` represents a column that has a `list` of `values`. for example:

`people = {
  "first": ['Corey', 'david', 'miaw'],
  "last": ['L1', 'L2', 'L3'],
  "email": ['e1@gm.com', 'e2@em.com', 'e3@pm.com']
}`

| Usage | Command | Description |
|:-----:|:-------:|:-----------:|
| Create DF with Dictionary | `df = pd.DataFrame(dict)` | `dict` is a python dictionary |
| Remove missing values | `df.drpna(axis=0, how=any, subset=None, inplace=False)` | `axis`: Determine if rows or columns which contain missing values to remove (0=`index` or 1=`columns`). `how`: determines remove condition, if {`any`, `all`} of the row/column is NoN then we remove it. `subset`: list of columns label (e.g. if you are dropping rows these would be a list of columns to include). |
| Drop Duplicates | `df.drop_duplicates(subset=None, keep='first', inplace=False)` | `subset`: {column label or sequence of labels, optional} Only consider certain columns for identifying duplicates, by default use all of the columns. `keep`: {`first`, `last`, `False`: Drop all duplicates.}, Determines which duplicates (if any) to keep. |
| Fill null values | `df.fillna(value, inplace=False)` | `value`: {scalar, dict, Series, or DataFrame}. Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame of values specifying which value to use for each index (for a Series) or column (for a DataFrame). Values not in the dict/Series/DataFrame will not be filled. This value cannot be a list. |

## Data Series
When we get values of a single column/row, pandas would call it a series. so if the output has multiple columns then it's a new filtered Data Frame.
| Usage | Command | Description |
|:-----:|:-------:|:-----------:|
| Counting Values | `df[column_name].value_counts(normalize=False)` | counts the number of occurrences of each different possible value for the column `column_name`. With `normalize` set to `True`, returns the relative frequency by dividing all values by the sum of values. |

## Accessing
In `iloc` we are searching with the **integer** location. but in `loc` we're searching by **labels/strings**.

**note**: We can always replace a list of numbers or column names with list-slicing equivalence. for example `df.loc[[3,4,5,6], 'email']` is the same as `df.loc[3:6, 'email']`.

**note 2**: if the index is not the default integer like 0, 1, 2,... then we can't use any number i in `df.loc[i]` for accessing the i'th row of DF. with the `loc` method we need to give the **index value** of row(s) so when index values are not integers we can't access them via numbers. but still `iloc` method is working here and has no problem. for example we can use `df.iloc[0]` for accessing first row.

| access what? | Command | Example or Description |
|:------------:|:-------:|:-----------:|
| A column | `df['name_of_column']` | `df['name']` |
| Multiple columns | `df[a_list_of_column_names]` | `df[['name', 'email']]` |
| A row | `df.iloc[i]` or `df.loc[i]` | `i` is the number of specific row |
| Multiple rows | `df.iloc[a_list_of_rows_numbers]` | `df.iloc[[0, 1, 2, 3]]` will return first 4 rows. also we can do this with `loc` too. |
| A single value in table | `df.loc[row_number, column_name]` | `df.loc[5, 'email']` |
| Multiple rows & columns | `df.loc[list_of_numbers, list_of_strings]` | `list_of_numbers` describes which rows we want and `list_of_strings` is list of desirable column's labels. |
| Multiple rows & columns with slicing |  `df.loc[i:j, label_start:label_end]` | slicing acts inclusive in pandas. `i` and `j` are integers. |


## Indexes
Whenever we want the changes to be permanent on the original DF we add `inplace=True`. without using this, it allows us to see what things look like without actually affecting DF itself.

**Defining the index while Loading DF**: `df = pd.read_csv(path, index_col='column_name')`

**Create a new DF with different column as Index**: `new_df = df.set_index('column_name')`

**Change the index of current DF**: `df.set_index('column_name', inplace=True)`

**Change the index to default integer index for current DF**: `df.reset_index(inplace=True)`

**Create a new DF rows sorted based on Index**: `df.sort_index(ascending=True)`


## Filtering
Using Conditionals to Filter Rows and Columns

**Create a Conditional Filter**: for example: `filt = df['close_price'] > 100`. this is a data series with a boolean value for each row.

**Create an IS IN Filter**: suppose we have a list of countries named `countries`. now we only want rows if 'Country' value of them is in the `countries` list. `filt = df['Country'].isin(countries)`.

**Create a Contains Filter**: suppose we have a column that shows each person has worked with what languages. in this column, there are many languages. we're looking to see whether the person has worked with `Python` or Not. So: `filt = df['LanguageWorkedWith'].str.contains('Python', na=False)`

**Apply the filter directly**: `df[filt]`. this will return the whole DF but filtered.

**Apply the filter with loc method**: `df[filt, column_list]`. this will return filtered DF with only specified columns. if there is no column name listed then it acts like `df[filt]`.

**AND/OR the Filters**: we can use `&` for AND operation and `|` for OR operation. example: `df[filt_1 & filt_2]`

**NOT the Filter**: `~`. example: `df.loc[~filt, 'volume']`


## Altering DF
Updating Rows and Columns - Modifying Data Within DataFrames
[click](https://www.youtube.com/watch?v=DCDe29sIKcE&list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS&index=5)

## Data Engineering

### 1 One-hot Encoding Categorical Features

One-hot encoding aims to transform a categorical variable with `n` possible outputs into `n` binary variables.

**data:** DataFrame to be used

**prefix:** A list with prefixes. String to append DataFrame column names.

**columns:** Column names in the DataFrame to be encoded. If `columns` is `None` then all the columns with object, string, or category dtype will be converted. `prefix` and `columns` must have the same length.

```python
# This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
df = pd.get_dummies(data = df, prefix = cat_variables, columns = cat_variables)
```

### 2 

