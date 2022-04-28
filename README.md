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
The Rest API is also deployed on heroku.

----
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
   * Enable automatic deployments that only deploy if your continuous integration passes.
   * Hint: think about how paths will differ in your local environment vs. on Heroku.
   * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Set up DVC on Heroku using the instructions contained in the starter directory.
* Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy`
* Write a script that uses the requests module to do one POST on your live API.
