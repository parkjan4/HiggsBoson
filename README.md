# Classification of Higgs Boson using Linear Methods
CS433 - Machine Learning, 2018, EPFL

This is the directory of the project for the course "Machine Learning" fall 2018, EPFL. This directory contains Python files for project implementation. For more detailed explanation of the project please refer to the report (`Report.pdf`). 

The project presents a challenge of correctly classifying Higgs bosons based on the 30-feature particle accelerator data from CERN. This is a classic binary classification task in which we are specifically investigating the strength of and the extent to which linear regression methods can achieve the task.

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

* `data/`. Directory containing original dataset from LINQS. [online] Linqs.soe.ucsc.edu. Available at: https://linqs.soe.ucsc.edu/node/236 [Accessed 11 Jan. 2019].

* `project/`. Folder containing the actual code files for the project:
    * `gephi/` Folder containing gephi files for visualization and exploration of the network.
    * `images/` Folder containing different images that are generated when running the different notebooks.
    * `fragmentation measures.py` Contains functions to compute fragmentation measures on the provided network.
    * `optimization_algorithms.py` Contains both optimization algorithms for fragmentation and information flow as well as the necessary functions to compute the respectives objective values. 
    * `data_exploration_functions.py` Contains several functions used for the import and parse of the data, creation of the network structure or identification of largest component among others.
    * `fragmentation.ipynb` Notebook containing initial data exploration as well as optimization task and results on the fragmentation problem. The provided notebook is already executed and shows the desired results.
    * `information_flow.ipynb` Notebook containing the data exploration and optimization task and results on the information diffusion problem. The provided notebook is already executed and shows the desired results. A new execution can take around 15 to 20 minutes. 
    * `adjacency.npy` Numpy file containing the structure of the adjacency matrix of the original network. Can be used to avoid creating it from scratch if a new execution of any of the two notebooks wants to be done. 

* `Report.pdf`
* `requirements.txt`


## How to execute the files.
	
Only fragmentation and information flow Notebooks are intended to be executed. All other files do not provide any directly readable result. The project has been developed so that fragmentation notebook is read first as it contains an initial exploration of the data. Nevertheless, information_flow notebook can be read and understood without need of previous consultation to the fragmentation notebook, taking into account the reader is aware of the purpose of the project.

## Authors

* **Jean-Baptiste Beau** 
* **Frédéric Myotte** 
* **Jangwon Park** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
