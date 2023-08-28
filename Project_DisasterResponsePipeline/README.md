# Disaster Response Pipeline Project

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [Components](#components)
 * [Instructions of How to Interact With Project](#instructions-of-how-to-interact-with-project)
 * [Licensing, Authors, Acknowledgements, etc.](#licensing-authors-acknowledgements-etc)
 
### Project Motivation
In this project, I applied my data engineering skills and machine learning technique to analyze disaster data from [Figure Eight](https://appen.com/). A web app has been built and the underlying model serves to classify text message into different categories. The automatic labeled message could be sent to disaster response department and thus could be helpful to early detect the nature of the disaster and help mitigate its consequence.

### File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # data cleaning pipeline    
|- InsertDatabaseName.db # database to save clean data to     


models   

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model     

img

|- screenshot1.jpg # web app screenshot   
|- screenshot2.jpg # web app screenshot

README.md    

### Components
process_data.py
train_classifier.py

#### 1. ETL Pipeline
A Python script, `process_data.py`, writes a data cleaning pipeline that:

 - Loads the messages and categories datasets in csv format
 - Preprocess and merges the two datasets 
 - Stores it in a SQLite database
 
A jupyter notebook `ETL Pipeline Preparation` was used to initially explore the data to prepare the process_data.py python script. 
 
#### 2. ML Pipeline
A Python script, `train_classifier.py`, writes a machine learning pipeline that:

 - Loads data from the SQLite database
 - Builds a NLP processing and machine learning pipeline
 - Trains and tunes a model using GridSearchCV
 - Outputs evaluation results on the test set
 - Exports the final model as a pickle file
 
A jupyter notebook `ML Pipeline Preparation` was used to explore possible machine learning methods to prepare the train_classifier.py python script. 

#### 3. Flask Web App
The project includes a web app where users can input a new message and get classification results in several categories. The web app will also display the distribution of message genres and categories in training set. The outputs are shown below:

![](https://github.com/zhukuixi/Udacity_DataScientistNanoDegree/blob/main/Project_DisasterResponsePipeline/img/screenshot1.jpg)  
![](https://github.com/zhukuixi/Udacity_DataScientistNanoDegree/blob/main/Project_DisasterResponsePipeline/img/screenshot2.jpg)



### Instructions of How to Interact With Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements, etc.
Thanks to Udacity for starter code for the web app. 
