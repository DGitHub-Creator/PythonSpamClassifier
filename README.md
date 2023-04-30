# Spam Classification System

The Spam Classifier is a machine learning model designed to classify emails as spam or non-spam. This system utilizes the TREC06P dataset and employs Naive Bayes and LSTM models to perform classification tasks.

## Getting Started

Follow these instructions to set up the project on your local machine for development and testing purposes.

### Prerequisites

To install the software, you will need the following:

- Python 3.9
- Pip
- Git
- Conda (Anaconda or Miniconda)
- Additional packages can be found in the `requirements.txt` file, and an `environment.yml` file is provided for convenience.

### Installation

1. Clone or download the repository to your local computer:

   ```bash
   git clone https://github.com/DGitHub-Creator/PythonSpamClassifier.git
   ```

2. Change to the project directory:

   ```bash
   cd PythonSpamClassifier
   ```

3. (Optional) Create a virtual environment using Conda:

   You can directly create a conda environment using the following command:

   ```bash
   conda env create -f environment.yml 
   ```

   Alternatively, create a conda environment using these commands:

   a. Create a Conda virtual environment called "tensorflow_env" with Python version 3.9:

   ```bash
   conda create --name tensorflow_env python=3.9
   ```

   b. Activate the virtual environment:

   ```bash
   conda activate tensorflow_env
   ```

   c. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

After setting up the environment and installing the required packages, activate the environment, go to the project directory and use the spam filter as follows:

```bash
python GUI.py
```

The graphical user interface (GUI) will appear, prompting you to enter your email. The emails will be classified as spam or non-spam based on the selected model, with results displayed in the GUI.

## Built With

- [Scikit-learn](https://scikit-learn.org/stable/) - A machine learning library for Python
- [Numpy](https://numpy.org/) - A library for working with arrays
- [Pandas](https://pandas.pydata.org/) - A library for working with dataframes
- [Tensorflow](https://www.tensorflow.org/) - An open-source software library for dataflow and differentiable programming
- [Keras](https://keras.io/) - An open-source software library for deep learning
- [PyQt5](https://pypi.org/project/PyQt5/) - A set of Python bindings for the Qt libraries for GUI development

## Contributing

If you would like to contribute to the project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://chat.openai.com/LICENSE.md) file for details.