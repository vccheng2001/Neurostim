from django.shortcuts import render
# django 
from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, Http404
from django.utils.translation import ugettext as _
from django.conf import settings

# user login/logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout

# forms 
from apnea_detection.forms import LoginForm, RegisterForm, UploadFileForm
from apnea_detection.models import UploadFile
import sys
import os
import glob
import json
from json import JSONEncoder
sys.path.append(os.path.abspath(os.path.join('../')))
print(f'Added {sys.path[-1]} to path')
from flatline_detection import FlatlineDetection


import pickle 
import pandas as pd
import os
import csv
from datetime import date, datetime
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from sklearn import linear_model 
from pathlib import Path
import re 
# plot
import plotly
from plotly.offline import plot
from plotly.graph_objs import Scatter
# disable debugging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# root directory is 
ROOT_DIR = Path(__file__).parents[2]
print("ROOT", ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
INFO_DIR = os.path.join(ROOT_DIR, "info")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")

import matplotlib
matplotlib.use('agg')



''' helper function to convert csv to html '''
def csv_to_html(file):
    df = pd.read_csv(file, header=None)
    html = df.to_html(classes='table table-striped table-bordered table-responsive table-sm')
    return html

@login_required
def home(request):
    
    context = {}
    return render(request, "apnea_detection/home.html", context=context)

def handle_uploaded_file(file, form):
    match = re.search(r'(.*)_(.*)_ex(\d)_sr(\d)_sc(\d+)', file.name)
    if match:
        return match.groups()
    else:
        return None

@login_required
def visualize(request):
    context = {}

    # if user submitted file 
    if request.method == 'POST':
        print('-----Generating visualization------')
        # Visualize uploaded file 
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # returns model instance that was created/updated 
            params  = form.save()

            match = handle_uploaded_file(request.FILES['file'], form)
            # find file in local directory 
            if match is not None:
                dataset, apnea_type, excerpt, sample_rate, scale_factor = match 

                # update parameters 
                params.dataset = dataset
                params.apnea_type = apnea_type
                params.excerpt = excerpt
                params.sample_rate = sample_rate
                params.scale_factor = scale_factor
                params.save()
 
            else:
                context["error_message"] = "Error: could not parse file. Make sure the file name is of the form \
                                            <dataset>_<apnea_type>_ex<excerpt>, sr<sample_rate>, sc<scale_factor>"
                return render(request, "apnea_detection/error.html", context=context)
            fd = FlatlineDetection(ROOT_DIR, dataset, apnea_type, excerpt, sample_rate, scale_factor)
            
            # visualize 
            fig = fd.visualize()

            # display original graph 
            orig_graph = fig.to_html(full_html=False, default_height=500, default_width=700)
            context['orig_graph'] = orig_graph
            context['params'] = params
            context['show_flatline_button'] = True
            new_form = UploadFileForm()
            context["new_form"] = new_form
            return render(request, "apnea_detection/visualize.html", context=context) 

    # else display blank upload form
    else:
        new_form = UploadFileForm()
        context["new_form"] = new_form

    return render(request, "apnea_detection/visualize.html", context=context) 


def flatline_detection(request):
    context = {}
    
    params = UploadFile.objects.latest('id') # or date_updated

    # if no file found
    if not params:
        context["error_message"] = "No file to run flatline detection on. Please return to previous step."
        return render(request, "apnea_detection/error.html", context=context) 
    
    # flatline detection 
    fd = FlatlineDetection(ROOT_DIR, params.dataset, params.apnea_type, \
        params.excerpt, params.sample_rate, params.scale_factor)


    print('------Running flatline detection-------')
    flatline_fig, flatline_times, nonflatline_fig, nonflatline_times = fd.annotate_events(15, 0.1, 0.95)
    flatline_fig = flatline_fig.to_html(full_html=False)
    nonflatline_fig = nonflatline_fig.to_html(full_html=False)

    # save as context
    context['flatline_fig'] = flatline_fig
    context['nonflatline_fig'] = nonflatline_fig
    context['num_flatline'] = len(flatline_times)
    context['num_nonflatline'] = len(nonflatline_times)
    context['params'] = params

    return render(request, "apnea_detection/flatline_detection.html", context=context) 



''' Normalizes a file specified by user '''
def normalize(form):
    # cleaned form 
    excerpt             = form["excerpt"]
    dataset             = form["dataset"]
    apnea_type          = form["apnea_type"]
    sample_rate         = form["sample_rate"]
    norm                = form["norm"]
    slope_threshold     = form["slope_threshold"]
    scale_factor_low    = form["scale_factor_low"]
    scale_factor_high   = form["scale_factor_high"]

    # read unnormalized file
    unnormalized_file = f"{DATA_DIR}/{dataset}/preprocessing/excerpt{excerpt}/filtered_{sample_rate}hz.txt"
    df = pd.read_csv(unnormalized_file, delimiter=',')

    # sample every nth row 
    # df = df.iloc[::100, :]

    # calculate slope of signal
    reg = linear_model.LinearRegression()
    reg.fit(df["Time"].values.reshape(-1,1),  df["Value"].values)
    slope = reg.coef_[0] * 1000  # convert unit
    print("Slope:", slope)
    
    # perform linear scaling 
    scale_factor = scale_factor_high if slope > slope_threshold else scale_factor_low
    df["Value"] *= scale_factor

    # write normalized output file
    normalized_file = unnormalized_file.split('.')[0] + f"_{norm}_{scale_factor}" + ".norm"
    df.to_csv(normalized_file, index=None, float_format='%.6f')
 
    # write new row to log.txt 
    log_file = f"{INFO_DIR}/log.csv"
    normalized_file_relpath = os.path.relpath(normalized_file, ROOT_DIR)
    with open(log_file, 'a', newline='\n') as logs:
        fieldnames = ['time','DB','patient','samplingRate','action','file_folder_Name','parameters']
        writer = csv.DictWriter(logs, fieldnames=fieldnames)
        print('Writing row....\n')
        time_format = '%m/%d/%Y %H:%M %p'
        writer.writerow({'time': datetime.now().strftime(time_format),
                        'DB': dataset,
                        'patient': excerpt,
                        'samplingRate': sample_rate,
                        'action': 'DataNormalization',
                        'file_folder_Name': normalized_file_relpath,
                        'parameters': f"slope:{slope_threshold}, hFactor:{scale_factor_high}, lFactor:{scale_factor_low}"})


    flatline_detection_params = dataset, apnea_type, excerpt, sample_rate, scale_factor 
    return normalized_file_relpath, flatline_detection_params


@login_required
def inference(request):
    context = {}
    results_file = f"{INFO_DIR}/summary_results.csv"

    # most recent preprocessing parameters
    preprocessing_params = Preprocessing.objects.last()
    print('preparams', preprocessing_params)


    try:
        test_acc = run_inference(preprocessing_params, test=True)
        # results = get_summary_results(results_file)
        # save model hyperparameters, display success message
        context["message"] = f"Successfully performed inference."
        context["results"] = results
        context["preprocessing_params"] = preprocessing_params
        context["test_acc"] = test_acc
        return render(request, "apnea_detection/inference.html", context=context)
            
            
    except Exception as error_message:
        # else throw error 
        context["error_heading"] = "Error during inference step. Please try again."
        context["error_message"] = error_message
        return render(request, "apnea_detection/error.html", context=context)
    return render(request, "apnea_detection/inference.html", context=context)



''' Retrieves results of latest run '''
def get_summary_results(results_file):
    result = pd.read_csv(results_file, index_col=None, squeeze=True)
    result = result.iloc[:,-8:].tail(1)
    results_dict = result.to_dict('r')[0]
    return results_dict


@login_required
def results(request):
    context = {}
    cols = ["dataset","apnea_type","excerpt","epochs", "batch_size","num_pos_train","num_neg_train",\
            "f1_1","f1_0","true_pos","true_neg","false_pos","false_neg"]

    # fieldnames = [  'dataset',      'apnea_type',  'excerpt', 'epochs', 'batch_size', 'num_pos_train',    'num_neg_train',\
    #                 'num_pos_test', 'num_neg_test', 'precision_1',  'precision_0',  'recall_1', 'recall_0',  'f1_1', 'f1_0',\
    #                 'true_pos','true_neg','false_pos','false_neg' ]

    # render results csv as html
    results_file = f"{INFO_DIR}/summary_results.csv"
    context["results"] = csv_to_html(results_file)
    context["preds"] = csv_to_html(os.path.join(ROOT_DIR, "sample_out.csv"))
    return render(request, "apnea_detection/results.html", context=context)

#####################################################################
#                     User Registration
####################################################################
def register_action(request):
    context = {}

    # If GET request, display blank registration form
    if request.method == 'GET':
        context['form'] = RegisterForm()
        return render(request, 'apnea_detection/register.html', context)

    # If POST request, validate the form
    form = RegisterForm(request.POST)
    context['form'] = form

    # Validates the form.
    if not form.is_valid():
        return render(request, 'apnea_detection/register.html', context)


    # Register and login new user.
    new_user = User.objects.create_user(username=form.cleaned_data['username'], 
                                        password=form.cleaned_data['password'],
                                        email=form.cleaned_data['email'],
                                        first_name=form.cleaned_data['first_name'],
                                        last_name=form.cleaned_data['last_name'])
    # Saves new user profile, authenticate 
    new_user.save()
    new_user = authenticate(username=form.cleaned_data['username'],
                            password=form.cleaned_data['password'])

    # Login
    login(request, new_user)
    return redirect(reverse('home'))

#####################################################################
#                     Logout action
####################################################################
def logout_action(request):
    logout(request)
    return redirect(reverse('login'))


#####################################################################
#                           Login
####################################################################
def login_action(request):

    context = {}

    # If GET request
    if request.method == 'GET':
        # If user already logged in, go to global stream page 
        if request.user.is_authenticated:
            return redirect(reverse('home'))  
        else:
            context['form'] = LoginForm()
            context["title"] = "Login"
            return render(request, 'apnea_detection/login.html', context)
        

    # If POST request, validate login form 
    form = LoginForm(request.POST)
    context['form'] = form

    # If not valid form 
    if not form.is_valid():
        print('Invalid login')
        return render(request, 'apnea_detection/login.html', context)

    # Authenticate user and log in
    username = form.cleaned_data['username']
    password = form.cleaned_data['password']
    user = authenticate(username=username, password=password)
    login(request, user)
    return redirect('/')