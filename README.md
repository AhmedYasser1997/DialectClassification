# DialectClassification

## Introduction
Many countries speak Arabic; however, each country has its own dialect, the aim of this task is to build a model that predicts the dialect given the text.
The Project is divided in the following Sections:

* Data Fetching: Given the ids of the text tweets, we make an API call by a POST request
* Data Processing: Wrangling the data and performing regular NLP cleaning such as special words and emoji removal
* A Machine Learning Pipeline to train a model able to classify text message to its dialect
* A web app that predicts the dialect of the text message given as an input message.

## Files Description
**dialect_dataset.csv:** CSV file that contains the ids of the texts and their corresponding dialect. \
**Data fetching notebook.ipynb** Loads the texts and merges them with the current data
**Data preprocessing notebook.ipynb** Data wrangling mainly and some EDA  \
**Model training notebook.ipynb:** Building 2 models: 1 ML model (Naive Bayes) and 1 Deep Learning model (using recurrent neural network) which unfortunately is still incomplete \
### App Folder
**run.py:** Python script for running the web app \
**go.html and master.html:** templates folder that contains 2 HTML files for the app front-end

## Running the project:
Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

## Summary 
The final output of the project is an interactive web app that takes a message from the user as an input and then classifies it.