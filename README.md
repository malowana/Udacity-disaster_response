# Udacity-disaster_response

<b>Table of Contents</b>
1. Description
2. Dependencies
3. Program execution
4. Screenshots


<b> 1. Description </b><br>
This project is a part of Udacity course. The aim of this project was to create disaster response clasificator.

In data/process_data.py I joined csv files and create database. <br>
In models/train_classifier.py I perform feature engineering, create ML pipeline and built model.  <br>
In app/run.py fronted has been created and model shared using Flask.  <br>
 <br>

<b> 2. Dependencies </b><br>
Needed libraries: <br>
NumPy, SciPy, Pandas, Sciki-Learn, NLTK, SQLalchemy, Pickle, Flask, Plotly <br>
<br>

<b> 3. Program execution </b><br>
a) clone git repository: <br>
<pre>
https://github.com/malowana/Udacity-disaster_response.git  
</pre>
 
b) You can run again command below to create db and train model again or skip this step and use uploaded model:<br>
 -To run ETL pipeline to clean data and store the processed data in the database 
  <pre>
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
</pre>
<br>
  -To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python <br>
  <pre>
models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
</pre>

c) Run your web app:   <br>
<pre>
python run.py
</pre>

d) Go to http://0.0.0.0:3001/ and play with the app <br>
<br>
<b> 4. Screenshots </b><br>
a) main page where is plot of distribution of genres and categories and place where you can put your message <br>
![This is an image](a1.JPG) <br><br>
b) after clicking "Classiby message" button you will be forwarded to the next page where your message will be classified to the proper category <br>
![This is an image](a2.JPG) <br><br>
