# Udacity MLOps Project 3: Census Data ML module

## Environment Setup
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * In order to install the package in development mode (e.g. needed for testing) run <br>``pip install -e .``


## Training process
If you have installed the package with ``pip install -e .`` you can use the the predefined entry point ``start_training`` otherwise run ``python training/train_model.py``

## Rest API
If you have installed the package with ``pip install -e .`` you can use the the predefined entry point ``run_rest`` otherwise switch to rest directory and run ``uvicorn app:app``.
You can find the documentation here: http://localhost:8000/docs
The Rest API is also deployed on heroku: https://census-ml.herokuapp.com/

Notice: Since heroku-github sync is currently not available due to security issues, the heroku git solution with two remotes was used (see https://status.heroku.com/incidents/2413 and https://stackoverflow.com/questions/71892543/heroku-and-github-items-could-not-be-retrieved-internal-server-error)
