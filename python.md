

# Managing packages
```bash
# where is the current pip? it should point to current active env
where pip

# checking python version
python --version

# To install a package
pip install `package-name`

# To install packages from a text file (requirements.txt)
pip install -r 'requirements.txt'

# Save current env packages versions to a text file:
pip freeze > 'requirements.txt'

# To upgrade a package
pip install --upgrade `package-name`

# To uninstall a package
pip uninstall `package-name`

# List of packages installed
pip list

# List of local packages
# these packages just relate to current venv and deleting them would not affect the global python
pip list --local
```



# Virtual evironments

## 1 VENV
for windows we use `py` but in LINUX/MAC we should use `python3`.

**1. Create a virtual environment:**

the new env will be saved in current directory. python version is same as global python version.
```bash
# new venv has just pip package
py -m venv 'NAME'
# all the global python packages are accessible in the new venv
py -m venv 'NAME --system-site-packages
```

**2. Delete a virtual environment:**

First deactivate the env and then just delete the env folder.
```bash
# windows
rmdir 'NAME' /s
# Linux
rm -fr 'NAME'
```

**3. Activate virtual environment:**

You should be in the directory where the env has saved.

Windows Powershell:
```bash
'NAME'\Scripts\Activate.ps1
```
Linux/MAC:
```bash
source 'NAME'/bin/activate
```

**4. Deactivate virtual environment:**

```
deactivate
```

## 2 Conda
works just in CMD.
### Create a New Environment
Specifying the python version in `conda create` will automatically include the default pip for that Python version and it is separate from the pip of base environment.
```python
# approach 1: specifying the name
conda create NAME python=python_version

# approach 2: specifying the path
# channels: free, conda-forge
conda --prefix "/path/to/custom/env" python=python_version --channel free

# approach 3: clone an existing environment
conda create --name NEW_ENV_NAME --clone BASE_ENV_NAME
```

### Working with Envs 
```python
# environments list
conda info --envs

# Activating an environment
activate NAME

# Deactivate an env
conda deactivate NAME
```
### Checking wheather the cuda is installed?
```python
# Install CUDA Packages
conda install -c conda-forge cudatoolkit=11.8 cudnn
conda env config vars set CUDA_HOME=$CONDA_PREFIX

# Verifying CUDA Installation
# Run the following command to see if CUDA packages are present:
conda list cudatoolkit
# or
nvcc --version
```


# Python Libraries

## Financial

### Visualization
| **Name** | **Description** | **Link(s)** |
|:--------:|:---------------:|:-----------:|
| mplfinance | Financial Markets Data Visualization using Matplotlib | [github](https://github.com/matplotlib/mplfinance), [PyPi](https://pypi.org/project/mplfinance/) |
### Technical Analysis
| **Name** | **Description** | **Link(s)** |
|:--------:|:---------------:|:-----------:|
| ta-lib-python | TA-Lib is widely used by trading software developers requiring to perform technical analysis of financial market data. Includes 150+ indicators such as ADX, MACD, RSI, Stochastic, Bollinger Bands, etc. Candlestick pattern recognition. Open-source API for C/C++, Java, Perl, Python and 100% Managed .NET | [Python Github](https://github.com/TA-Lib/ta-lib-python), [Documentation](http://ta-lib.github.io/ta-lib-python/), [WebSite](https://ta-lib.org/) |
| pandas-ta | Technical Analysis Indicators - Pandas TA is an easy to use Python 3 Pandas Extension with 150+ Indicators | [Github](https://github.com/twopirllc/pandas-ta), [Documentation](https://twopirllc.github.io/pandas-ta/) |
| TA | It is a Technical Analysis library useful to do feature engineering from financial time series datasets (Open, Close, High, Low, Volume). It is built on Pandas and Numpy. | [Github](https://github.com/bukosabino/ta), [Documentation](https://technical-analysis-library-in-python.readthedocs.io/en/latest/) |
| talipp | talipp (or tali++) is a Python library implementing financial indicators for technical analysis. The distinctive feature of the library is its incremental computation which fits extremely well real-time applications or applications with iterative input in general. | [Github](https://github.com/nardew/talipp), [Documentation](https://nardew.github.io/talipp/latest/) |
| finta | Common financial technical indicators implemented in Pandas. | [github](https://github.com/peerchemist/finta) |
