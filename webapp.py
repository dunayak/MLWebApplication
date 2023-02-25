import numpy as np
import pickle
import os
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from flask import Flask, request, jsonify, render_template, send_file,flash,redirect,url_for
from flask_caching import Cache
from sklearn import metrics 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

UPLOAD_FOLDER = 'static/Uploadedfiles'
ALLOWED_EXTENSIONS = {'csv'}

webapp = Flask(__name__)  # Initialize the flask webApp
webapp.config['SECRET_KEY'] = 'super secret key'
webapp.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
webapp.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
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

@webapp.route('/loaddata',methods=['GET','POST'])
def loaddata():
    if request.method == 'POST':
        honeyfile = request.files['file']
        filename = honeyfile.filename

        print('Honeydata',filename)
        honeyfile.save(os.path.join(filename))
        show_pred_app=True
        cache.set('testcsv_file_name',filename)
        return render_template('indexflask.html',outcome='Uploaded',show_pred_app=show_pred_app)

@webapp.route('/processTest',methods=['GET','POST'])
def processTest():
    myModel = 'static/Models/PCAKNN.pkl'
    # Getting filename from cache, gathered from UI/loaddata route
    DSFileNamewithPath=f"static/Uploadedfiles/{cache.get('testcsv_file_name')}"

    if cache.get('testcsv_file_name') == None:
        flash('No File Selected')
        return redirect(url_for('home'))

    print(f"Clearing Cache file name {cache.get('testcsv_file_name')}")
    cache.clear()
    print(f"File Location to load {DSFileNamewithPath}")
    honeydata_df = pd.read_csv(DSFileNamewithPath)
    # Data cleaning
    honeydata_df=honeydata_df.drop('label',axis=1) # Dropping label columns as Honey Label provides specific information
    honeydata_df["label.full"]=honeydata_df["label.full"].map(str.upper)
    # Renaming column name to meaningful name 
    honeydata=honeydata_df.rename(columns={"label.full": "Honey Label"})
    label_full = honeydata["Honey Label"].copy()
    honey_no_labels=honeydata.iloc[ :,:700].copy()
    # Data Transformation/ Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(honey_no_labels, label_full,test_size=0.20,stratify=label_full,random_state=1)

    X_train=X_train.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)
    X_test=X_test.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)

# Preprocessing using Principal Component Analyses  (PCA)  
    pca = PCA(n_components=3)
    X_train_pca=pca.fit_transform(X_train)
    X_test_pca=pca.transform(X_test)
    
    pca_test_df = pd.DataFrame(data = X_test_pca,
                     columns = ["PC1", 
                                "PC2",
                                "PC3"
                                ])
    pca_test_df=pca_test_df.reset_index(drop=True)
    
    
    mlmodel = pickle.load(open(myModel, 'rb'))
    y_pred = mlmodel.predict(pca_test_df)
    test_set_accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted',zero_division=0)
    recall = recall_score(y_test, y_pred,average='weighted')
    f1 = f1_score(y_test, y_pred,average='weighted')

    data=[{"Model":"PCA With KNN","Accuracy":test_set_accuracy,"Precision":precision,"Recall":recall,"F1":f1}]

    perform_matrix=pd.DataFrame(data)
    show_pred_app=True
    return render_template('indexflask.html',perform_matrix=[perform_matrix.to_html()],fields=[''],show_pred_app=show_pred_app )

@webapp.route('/plot_confusion')
def plot_confusion():
    graph_file=None
    graph_file=f"static/Graphs/{cache.get('global_selected_algo')}.png"
    return send_file(graph_file, mimetype='image/png')

@webapp.route('/knn_confusion')
def knn_confusion():
    knngraph_file=None
    knngraph_file=f"static/Graphs/PCAKNN.png"
    return send_file(knngraph_file, mimetype='image/png')

if __name__ == "__main__":
    webapp.run(host='0.0.0.0', port=80, debug=True)
