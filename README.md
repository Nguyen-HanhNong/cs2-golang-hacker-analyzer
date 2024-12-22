# cs2-golang-hacker-analyzer

## Project Description

This project is a tool that can be used to determine if a player is cheating in the game Counter-Strike: Global Offensive. The tool uses machine learning models to analyze the player's statistics and determine if they are cheating. The tool takes the player's Steam ID as input and outputs whether or not the player is cheating. It uses Golang to scrape the important data from the player's profile and Python to analyze the data using machine learning models.

## Languages/Libraries used in the project
- [Playwright](https://playwright.dev/python/docs/intro) - Playwright is a Python library that allows you to automate web browsers.
- [Numpy](https://numpy.org/) - NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- [Pandas](https://pandas.pydata.org/) - Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and data manipulation library built on top of the Python programming language.
- [Scikit-learn](https://scikit-learn.org/stable/) - Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression, and clustering algorithms including support vector machines, random forests, gradient boosting, k-means, and DBSCAN.
- [Tensorflow](https://www.tensorflow.org/) - TensorFlow is an open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library and is also used for machine learning applications such as neural networks.
- [SQLite](https://www.sqlite.org/index.html) - SQLite is a C-language library that implements a small, fast, self-contained, high-reliability, full-featured, SQL database engine.
- [Golang](https://golang.org/) - Go is an open-source programming language that makes it easy to build simple, reliable, and efficient software.
- [Python](https://www.python.org/) - Python is an interpreted high-level general-purpose programming language.
- [Matplotlib](https://matplotlib.org/) - Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

## Steps to run the project
1. Install python using the following [link](https://www.python.org/downloads/).
2. Install the required packages from the following list:
- ```pip install pytest-playwright```
- ```playwright install```
- ```pip install numpy```
- ```pip install scikit-learn```
- ```pip install tensorflow```
3. Clone the project using the following command: ```git clone https://github.com/Nguyen-HanhNog/cs2-golang-hacker-analyzer.git```
4. Navigate to the project directory using the following command: ```cd cs2-golang-hacker-analyzer```
5. Create the database by running the following command: ```python createDatabase.py```
6. Create and train the models by running the following command: ```python trainModels.py```
7. Then, when you want to check if a player is cheating, run the following command: ```python main.py <steamID>```, where steamID is the steamID of the player you want to check.