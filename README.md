# Classification of Higgs Boson using Linear Methods
CS433 - Machine Learning, 2018, EPFL

This is the directory of the project for the course "Machine Learning" fall 2018, EPFL. This directory contains Python files for project implementation. For more detailed explanation of the project please refer to the report (`Report.pdf`). 

The project presents a challenge of correctly classifying Higgs bosons based on the 30-feature particle accelerator data from CERN. This is a classic binary classification task, in which we are specifically investigating the strength of and the extent to which linear regression methods can achieve the task. 

It is worth noting that all machine learning algorithms used in the task are developed from scratch. This was to learn the mechanics of the algorithms first-hand to demonstrate a good working understanding and knowledge of machine learning. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The required environment for running the code and reproducing the results is a computer with a valid installation of Python 3. More specifically, [Python 3.6](https://docs.python.org/3.6/) is used.

Besides that (and the built-in Python libraries), the following packages are used and have to be installed:

* [NumPy 1.13.3](http://www.numpy.org). `pip3 install --user numpy==1.13.3`
* [Matplotlib 2.0.2](https://matplotlib.org). `pip3 install --user matplotlib==2.0.2`
* [Pandas 0.23.4](https://pandas.pydata.org)    `pip install --user pandas==0.23.4`

### Installing

To install the previously mentioned libraries a requirements.txt file is provided. The user is free to use it for installing the previously mentioned libraries.  

## Project Structure

The project has the following folder (and file) structure:

* `data/`. Contains original train and test dataset from CERN.

* `Python Code/`. Folder containing the actual code files for the project:
    * `implementations.py` All functions related to model building and classification.
    * `helpers.py` All other functions including data loading and exporting predictions as .csv.
    * `run.py` Imports and runs classification model automatically upon calling. Generates our best results as .csv.

* `Report.pdf`
* `requirements.txt`


## How to execute the files.
	
From a console, you should be able to get the csv file that gave us our best result by running the following command (in the directory of run.py file): "python run.py". Be aware of having the correct path to load train and test files.

## Authors

* **Jean-Baptiste Beau** 
* **Frédéric Myotte** 
* **Jangwon Park** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
