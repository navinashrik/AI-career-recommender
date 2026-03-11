#  AI Career & Higher Studies Recommendation System

I built this project to help students figure out which career or higher study path suits them best. A lot of students including myself struggle with this decision after school so I thought why not let a machine learning model help with it.

## What it does

You enter some basic details about yourself like your academic scores in subjects like maths, science, arts and commerce, a few of your skills and some personality traits. The system then runs that through a trained ML model and tells you which career path fits you the most along with a confidence percentage for each option.

The careers it can currently recommend are:
- Engineering
- Computer Science
- Business
- Arts & Design


## How I built it

I used Python and a few popular libraries to build this. The model is trained on student data stored in a CSV file. I tried both a Decision Tree and a Random Forest classifier and went with whichever gave better accuracy.

For the web interface I used Streamlit which made it really easy to turn the Python script into something anyone can use without touching any code.

## Tech stack

- Python
- pandas
- scikit-learn
- matplotlib
- Streamlit
- pickle

## How to run it

First install the required libraries:
pip install pandas scikit-learn matplotlib streamlit

Then retrain the model by running:
py career_recommender.py

Then launch the app:
py -m streamlit run app.py

## Project structure


career-recommender/
├── career_recommender.py   # trains and saves the ML model
├── app.py                  # streamlit web app
├── students.csv            # dataset used for training
├── career_model.pkl        # saved trained model
├── label_encoder.pkl       # saved label encoder
└── README.md               # this file


## What I learned

This was my first proper machine learning project. I learned how to prepare data, train ML models, evaluate their accuracy and build a simple web app around it. It was challenging at times especially during training the ML model. But with better understanding and a clear goal i was able to complete this project sucessfully!

Feel free to fork it, improve the dataset or add more careers to it!
