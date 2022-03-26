# example of using Flask to deploy a fastai deep learning model trained on a tabular dataset
import json
import os
import urllib.request
import numpy as np
import pandas as pd
import pathlib
import pickle
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request

input_scoring_columns = ['season','temp','humidity']
scoring_columns = ['temp', 'season_2', 'season_3', 'season_4', 'humidity']

# load the trained model
model_file = 'bike_model.sav'
loaded_model = pickle.load(open(model_file, 'rb'))
#path = Path(os.getcwd())
#full_path = os.path.join(path,'adult_sample_model.pkl')
#print("path is:",path)
#print("full_path is: ",full_path)
# load the model
#learner = load_learner(full_path)


app = Flask(__name__)


@app.route('/')
def home():   
    ''' render home.html - page that is served at localhost that allows user to enter model scoring parameters'''
    title_text = "fastai deployment"
    title = {'titlename':title_text}
    return render_template('home.html',title=title) 
    
@app.route('/show-prediction/')
def show_prediction():
    ''' 
    get the scoring parameters entered in home.html and render show-prediction.html
    '''
    # the scoring parameters are sent to this page as parameters on the URL link from home.html
    # load the scoring parameter values into a dictionary indexed by the column names expected by the model
    score_values_dict = {}
    # bring the URL argument values into a Python dictionary
    for column in input_scoring_columns:
        # use input from home.html for scoring
        if column == "season":
            # ["Winter" ,"Spring" ,"Summer" ,"Fall"]
            # feature_cols = ['temp', 'season_2', 'season_3', 'season_4', 'humidity']
            if request.args.get(column) == "Winter":
                score_values_dict['season_2'] = 0
                score_values_dict['season_3'] = 0
                score_values_dict['season_4'] = 0
            elif request.args.get(column) == "Spring":
                score_values_dict['season_2'] = 1
                score_values_dict['season_3'] = 0
                score_values_dict['season_4'] = 0   
            elif request.args.get(column) == "Summer":  
                score_values_dict['season_2'] = 0
                score_values_dict['season_3'] = 1
                score_values_dict['season_4'] = 0   
            else:
                score_values_dict['season_2'] = 0
                score_values_dict['season_3'] = 0
                score_values_dict['season_4'] = 1
        else:
            score_values_dict[column] = request.args.get(column)
    for value in score_values_dict:
        print("value for "+value+" is: "+str(score_values_dict[value]))
    # create and load scoring parameters dataframe (containing the scoring parameters)that will be fed into the pipelines
    score_df = pd.DataFrame(columns=scoring_columns)
    # df = df.astype({"a": int, "b": complex})
    print("score_df before load is "+str(score_df))
    for col in scoring_columns:
        score_df.at[0,col] = score_values_dict[col]
    print("score_df: ",score_df)
    print("score_df.dtypes: ",score_df.dtypes)
    print("score_df.iloc[0]",score_df.iloc[0])
    print("shape of score_df.iloc[0] is: ",score_df.iloc[0].shape)
    # pred_class,pred_idx,outputs = loaded_model.predict(score_df.iloc[0])
    pred_bikes = loaded_model.predict(score_df)
    pred_bikes_str =  f"{pred_bikes[0]:.1f}"
    predict_string = "Predicted number of bike rentals is: "+pred_bikes_str
    # build parameter to pass on to show-prediction.html
    prediction = {'prediction_key':predict_string}
    # render the page that will show the prediction
    return(render_template('show-prediction.html',prediction=prediction))
    
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')