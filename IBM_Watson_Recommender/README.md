# IBM Watson Recommendation System

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [Web App](#gradio-Web-App)
 
### Project Motivation

In this project, I utilized my expertise in data science and machine learning to examine user-article interaction data from the IBM Watson Community. I developed a web application with an integrated model designed to recommend articles based on various user queries. Specifically, the application offers users three distinct functionalities: they can input their user ID to receive personalized article recommendations; alternatively, they can access a list of currently popular articles; or, they can input a specific term, prompting the system to suggest popular articles related to that term. This approach ensures a tailored and dynamic user experience, catering to individual preferences and interests.

### File Descriptions
|- app.py # the code to implement gradio frontend app  
|- Recommendations\_with\_IBM.ipynb # the jupyternote book for EDA  
|- recommender.py # the recommender system intergarted CF,ranked and content based recommender system  
|- recommmender\_CF.py # collaborative filtering recommender  
|- recommender\_content_based.py # content based recommender  
|- recommender\_ranked_based.py # ranked based recommender   
|- README.md # readme file  
|- requirements.txt # package requirement file  


data    

|- articles_community.csv # the data of articles  
|- user-item-interactions.csv # the user-article interaction data  



img

|- demo.png # web app screenshot     


    


#### Gradio Web App
The project includes a web app where users can get recommendation articles. 

[Link to Web App](https://huggingface.co/spaces/jooo/IBM_Watson_Recommendation)  

![](https://github.com/zhukuixi/Udacity_DataScientistNanoDegree/blob/main/IBM_Watson_Recommender/img/demo.png?raw=true)  



