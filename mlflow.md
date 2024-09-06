# MLflow

```python
import mlflow
```

## Experiments

| Description | Code |
|:-:|:-:|
| Create new Experiment | `mlflow.create_experiment(NAME)` |
| Tag new Experiment | `mlflow.set_experiment_tag(key, value)` |
| Set the Experiment | `mlflow.set_experiment(NAME)` |
| Delete Experiment | `mlflow.delete_experiment(NAME)` |


## Runs

| Description | Code |
|:-:|:-:|
| Start a run | `run = mlflow.start_run()` |
| run info | `run.info` |
| End a run | `mlflow.end_run()` |

| Logging to mlflow tracking | Code |
|:-:|:-:|
| a parameter | `mlflow.log_param(key, value)` |
| multiple parameters | `mlflow.log_params(dictonary)` |
| a metric | `mlflow.log_metric(key, value)` |
| multiple metrics | `mlflow.log_metrics(dictonary)` |
| a artifact | `mlflow.log_artifact(file)` |
| multiple artifacts | `mlflow.log_artifacts(directory)` |