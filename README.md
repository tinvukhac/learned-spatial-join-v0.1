# Deep learning based model for cost estimation of spatial join
## Required Environment

* We recommend to use [PyCharm](https://www.jetbrains.com/pycharm/download/) as the IDE. 
But you could use other IDEs(e.g. IntelliJ) or any other code editors.
* In order to make it easier for you to install all required libraries (Keras, TensorFlow, scikit-learn, pandas, etc), 
we would recommend you to install [Anaconda](https://docs.anaconda.com/anaconda/install/). In particular, you could use an environment which is identical with ours as the following steps:
1. [Install Anaconda](https://docs.continuum.io/anaconda/install/)
2. Add conda to your $PATH variable: /home/your_username/anaconda3/condabin
3. Move to the project directory: cd */deep-spatial-join
4. Follow this tutorial to create an environment from our environment.yml file: [Creating an environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
5. Activate the environment. Now you are ready to play with the model!  

## Brief description of the source code
* train.py: the endpoint to run the program.
* models.py: the implementation of the models which support train/test activity.
* datasets.py: data pre-processing module
* data/histograms: contains csv files, which are the histograms of input datasets (to be fed into the CNN layers).
* data/tabular: contains csv files, which are the tabular feature of the input datasets (to be fed into the MLP layer).
* data/join_results: contains csv files which are the results of spatial join queries. Columns: dataset 1, dataset 2, join result size, # of MBR test, execution time.
* trained_models: where you save the trained models.
* utils: a bunch of scripts that we use to clean/fix data problems. You do not need to pay much of attention to these scripts.  

## Train and test the models
* Train a model (then test):
```python
python train.py --tab data/tabular/tabular_all.csv --hist data/histograms/ --result data/join_results/train/join_results_small_x_small_uniform.csv --model trained_models/model_uniform.h5 --weights trained_models/model_weights_uniform.h5 --train
```
* Test a model (no train):
```python
python train.py --tab data/tabular/tabular_all.csv --hist data/histograms/ --result data/join_results/train/join_results_small_x_small_uniform.csv --model trained_models/model_uniform.h5 --weights trained_models/model_weights_uniform.h5 --no-train
```

## How to modify the current implementation?
* You can change the parameters, add/remove layers of MLP and CNN model at the function *create_mlp* and *create_cnn* in the [models.py](models.py) module.
* You can train/test in a specific group of datasets by using the corresponding join results at data/join_results.
* What if you want to train/test with your own data?
1. Run the join queries.
2. Export input dataset's histograms.
3. Export input dataset's spatial descriptors.
4. Make sure that the training/testing data files are in correct format (refer to existing files).
5. Train your own models. 
