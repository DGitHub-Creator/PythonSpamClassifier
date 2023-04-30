
Spam Classification System
===========

The Spam Classifier is a machine learning model for classifying emails as spam or non-spam. The model is trained on the TREC06P dataset and uses Naive Bayes and LSTM models to accomplish the classification task.

Getting Started
---------------

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

*   Python 3.9
*   Pip
*   Git
*   Conda (Anaconda  or Miniconda)
*   Other packages are detailed in requirements.txt, and of course I provide environment.yml

### Installing

1.  Clone or download the repository to the local computer:

```bash
git clone https://github.com/DGitHub-Creator/PythonSpamClassifier.git
```

2.  Change into the project directory:

```bash
cd PythonSpamClassifier
```

3. Create a virtual environment using Conda (optional)

   First, you can create a conda environment directly using the following command:

```bash
conda env create -f environment.yml 
```

​	Or, create a conda environment using these commands:

​	（1）Create a Conda virtual environment called "tensorflow_env", using Python version 3.9

```bash
conda create --name tensorflow_env python=3.9
```

​	（2）Activate the virtual environment:

```bash
conda activate tensorflow_env
```

​	（3）Install the required packages:

```
pip install -r requirements.txt
```

### Usage

Once the environment is installed, i.e. the required packages are installed, and after activating the environment, you can use the spam filter as follows:

```bash
python GUI.py
```

The GUI will display and guide you to enter your email. The emails are then classified as spam or non-spam based on the model you selected and the results are output to the GUI display.

Built With
----------

*   [Scikit-learn](https://scikit-learn.org/stable/) - A machine learning library for Python
*   [Numpy](https://numpy.org/) - A library for working with arrays
*   [Pandas](https://pandas.pydata.org/) - A library for working with dataframes
*   [Tensorflow](https://www.tensorflow.org/) - An open-source software library for dataflow and differentiable programming
*   [Keras](https://keras.io/) - An open-source software library for deep learning
*   [PyQt5](https://pypi.org/project/PyQt5/) - A set of Python bindings for the Qt libraries for GUI development.

Contributing
------------

If you would like to contribute to the project, please fork the repository and submit a pull request.

License
-------

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.