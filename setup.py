import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="census",
    author="datacamp489",
    version="0.1.0",
    description="ML module for training census data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires="==3.8.*",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "requests",
        "fastapi==0.63.0",
        "uvicorn",
        "gunicorn"
    ],
    entry_points={"console_scripts": [
        "start_training=training.train_model:main"]},

)
