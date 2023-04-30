
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
*   Conda (Anaconda  or Miniconda)
*   Other packages are detailed in requirements.txt, and of course I provide environment.yml

### Installing

1.  Clone the repository to your local machine:

```bash
git clone https://github.com/DGitHub-Creator/PythonSpamClassifier.git
```

2.  Change into the project directory:

```bash
cd Spam-Filter
```

3.  Create a virtual environment (optional):

```bash
python3 -m venv env
```

4.  Activate the virtual environment:

```bash
source env/bin/activate
```

5.  Install the required packages:

```
pip install -r requirements.txt
```

### Usage

Once you have the required packages installed, you can use the spam filter as follows:

```bash
python spam_filter.py path/to/email.txt
```

The model will classify the email as either spam or not spam, and print the result to the console.

Built With
----------

*   [Scikit-learn](https://scikit-learn.org/stable/) - A machine learning library for Python
*   [Numpy](https://numpy.org/) - A library for working with arrays
*   [Pandas](https://pandas.pydata.org/) - A library for working with dataframes

Contributing
------------

If you would like to contribute to the project, please fork the repository and submit a pull request.

License
-------

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.