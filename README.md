# Project IFT712

This is the repository of the IFT712 module final project @ UdeS.

It implements various machine learning models to classify the leaf dataset from [Kaggle](https://www.kaggle.com/competitions/leaf-classification) using sklearn:

- AdaBoost
- Bagging
- Decision Tree
- KNN
- LDA
- LinearSVC
- Logistic regression
- QDA
- Random forest
- SGD
- SVC

The notebook contains the implementation using the practical interface allowed by the python modules in the ```src``` folder. This interface is used to train, search the best hyper-parameters and get prediction metrics. Here is the format to respect:

```python
{
    # sklearn classifier name (case sensitive !)
    "name": str,
     # a dict containing the parameters of the init function of the selected classifier
     # (need to match the sklearn class object init function)
    "config": dict, (optional)
    # to standardize, need to be set, otherwise an error is raised
    "preprocess": bool,
    # for feature selection or reduction
    "feature" : {
        # the feature method : "reduction" or "selection" (case sensitive !).
        # If not set correctly, an error is raised
        "option": str,
        # a dict containing the parameters of :
        # sklearn PCA function if option was set to reduction
        # sklear SelectFromModel function if option was set to selection
        "config": dict
    }, (optional)
    # a dict containing the fit strategy, need to be set, otherwise an error is raised
    "fitStrategy": {
        # the validation method : "GridSearch" or "CV" (case sensitive !).
        # If not set correctly, an error is raised
        "option": str,
        # a dict containing the parameters of :
        # sklearn cross_validate function if option was set to CV
        # sklear GridSearchCV function if option was set to GridSearch
        # Also, need at least the scoring parameter to be set
        # otherwise, an error is raised
        "config": dict
    }
}
```

Finally, the report is in the ```doc``` folder.

## Disclaimer

The implementation is based on very recent packages version. Use an environnement to install them using the ```requirements.txt``` file:

```shell

pip install -r requirements.txt

```

## Authors

- Tahar Adam AMAIRI
- Corentin POMMELEC
- Chaimae Chaimae TOUBALI
