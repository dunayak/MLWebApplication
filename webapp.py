import numpy as np
import pickle
import os
import pandas as pd 
from flask import Flask, request, jsonify, render_template, send_file
from flask_caching import Cache
from sklearn import metrics 

webapp = Flask(__name__)  # Initialize the flask webApp

# Declaring the cache object, to be used to use the results between the different routes
cache = Cache(webapp, config={'CACHE_TYPE': 'simple'})

global_honey_dim_red_ds=None
global_selected_algo=None
global_full_test_df=None

@webapp.route('/',methods=['POST', 'GET'])
def home():
    # Rendering index file
    return render_template('indexflask.html')

@webapp.route('/resume.html')
def resume():
    return render_template('resume.html')


#show data to be selected by user

@webapp.route('/show_rows',  methods=("POST", "GET"))
def showRows():

    honey_df = pd.read_csv('static/data/pca_test.csv',index_col=0)
    cache.set('global_full_test_df',honey_df)

    return render_template('indexflask.html', features=[''],honeydata=[honey_df.to_html()])

# showHoney is to populate the honey sample data from preexisting files to the flaskindex page in form of HTML table
@webapp.route('/showHoney',  methods=("POST", "GET"))
def showHoney():
    # cache.clear()
    # selected_algo=None
    # for algo in request.form.values():
    #     selected_algo=algo
    honey_df = pd.read_csv('static/data/pca_test.csv',index_col=0)
    min=1
    max=honey_df.shape[0]
# Honey sample data record is being picked randomly from the preexisting honey sample csv file for prediction
    random_index = np.random.randint(min,max)
    honey_df_random=pd.DataFrame(honey_df.iloc[random_index-1:random_index])
# Saving results to cache so that other functions (routes) can use
    cache.set('global_honey_dim_red_ds',honey_df_random)

    return render_template('indexflask.html', sample_data=[honey_df_random.to_html()],fields=[''])
 
# predict route will predict the lable using selected algorithm by the used from User Interface 
@webapp.route('/predict',methods=['POST'])
def predict():
    random=False
    # cache.clear()
    # selected_algo=None
    
    if request.form.get("rowid") != None:
        rowid = int(request.form.get("rowid"))
        if rowid<0 or rowid>95:
           rowid=0 
        honey_full_df=cache.get('global_full_test_df')
        honey_test_df=honey_full_df.iloc[rowid:rowid+1]
        target=honey_test_df.iloc[:,-1]
    else:
        honey_test_df=cache.get('global_honey_dim_red_ds')
        target=cache.get('global_honey_dim_red_ds').iloc[:,-1]
        random=True

    selected_algo = request.form.get("Algo")
    cache.set('global_selected_algo', selected_algo)    
    selected_model=f"static/Models/{cache.get('global_selected_algo')}.pkl"

    honey_test_df_with_target=honey_test_df
    honey_test_df=honey_test_df.iloc[:,:3]
    mlmodel = pickle.load(open(selected_model, 'rb'))
    prediction = mlmodel.predict(honey_test_df)
    accuracy = metrics.accuracy_score(target, prediction)
    prediction=prediction[0]
    graph_file=f"static/Graphs/{cache.get('global_selected_algo')}.png"
    selected_algo=cache.get('global_selected_algo')
    return render_template('indexflask.html',sample_data=[honey_test_df_with_target.to_html()],fields=[''],selected_algo=cache.get('global_selected_algo'),accuracy=accuracy,honey_label=f'Predicted Honey Label: {prediction} using {selected_algo}', graph_file=graph_file,random=random)

@webapp.route('/plot_confusion')
def plot_confusion():
    graph_file=None
    graph_file=f"static/Graphs/{cache.get('global_selected_algo')}.png"
    return send_file(graph_file, mimetype='image/png')
if __name__ == "__main__":
    webapp.run(host='0.0.0.0', port=80, debug=True)
