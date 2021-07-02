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
from apnea_detection.forms import LoginForm, RegisterForm, UploadFileForm, FlatlineDetectionParamsForm
from apnea_detection.models import UploadFile, FlatlineDetectionParams
import sys
import os
import glob
import json
from json import JSONEncoder
sys.path.append(os.path.abspath(os.path.join('../')))
print(f'Added {sys.path[-1]} to path')
from flatline_detection import FlatlineDetection
from lstm import LSTM


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
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
import matplotlib
matplotlib.use('agg')



''' helper function to convert csv to html '''
def csv_to_html(file):
    df = pd.read_csv(file, header=None)
    html = df.to_html(classes='table table-striped table-bordered table-responsive table-sm')
    return html


#####################################################################
#                     Home page
####################################################################
@login_required
def home(request):
    
    context = {}
    return render(request, "apnea_detection/home.html", context=context)

def handle_uploaded_file(file, form):
    match = re.search(r'(.*)_(.*)_ex(.+)_sr(\d+)_sc(\d+)', file.name)
    if match:
        return match.groups()
    else:
        return None


#####################################################################
#                    Visualize uploaded file
####################################################################
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
                                            <dataset>_<apnea_type>_ex<excerpt>_sr<sample_rate>_sc<scale_factor>"
                return render(request, "apnea_detection/error.html", context=context)
            fd = FlatlineDetection(ROOT_DIR, dataset, apnea_type, excerpt, sample_rate, scale_factor)
            
            # visualize 
            fig = fd.visualize()

            # display original graph 
            orig_graph = fig.to_html(full_html=False, default_height=500, default_width=700)
            context['orig_graph'] = orig_graph
            context['params'] = params
            context['show_flatline_button'] = True

            context["upload_file_form"] = UploadFileForm()
            context["flatline_params_form"] = FlatlineDetectionParamsForm()
            return render(request, "apnea_detection/visualize.html", context=context) 

    # else display blank upload form
    else:
        upload_file_form= UploadFileForm()
        context["upload_file_form"] = upload_file_form

    return render(request, "apnea_detection/visualize.html", context=context) 



#####################################################################
#                     Flatline Detection
####################################################################

def flatline_detection(request):

    if request.method == "POST":

        if request.POST.get('flatline_thresh'):
            flatline_params_form = FlatlineDetectionParamsForm(request.POST)
            ft = float(request.POST.get('flatline_thresh'))
            lt = float(request.POST.get('low_thresh'))
            ht = float(request.POST.get('high_thresh'))
        else:
            flatline_params_form = FlatlineDetectionParamsForm()
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
    flatline_fig, flatline_times, nonflatline_fig, nonflatline_times = fd.annotate_events(ft, lt, ht)
    flatline_fig = flatline_fig.to_html(full_html=False)
    nonflatline_fig = nonflatline_fig.to_html(full_html=False)

    fd.output_apnea_files(flatline_times, nonflatline_times)

    # save as context
    context['flatline_fig'] = flatline_fig
    context['nonflatline_fig'] = nonflatline_fig
    context['num_flatline'] = len(flatline_times)
    context['num_nonflatline'] = len(nonflatline_times)
    context['params'] = params
    context['flatline_params_form'] = flatline_params_form


    return render(request, "apnea_detection/flatline_detection.html", context=context) 



#####################################################################
#                     Train/test
####################################################################
def train_test(request):
    context = {}
    params = UploadFile.objects.latest('id') 
    # 
    if request.method == "POST":

        if request.POST.get('test'):
            model = LSTM(ROOT_DIR, params.dataset, params.apnea_type, params.excerpt, 64, 30)
            test_error = model.test()
            context['message'] = f"Final test error: {test_error}"
            return render(request, "apnea_detection/train_test.html", context=context) 
        else:
            model = LSTM(ROOT_DIR, params.dataset, params.apnea_type, params.excerpt, 64, 30)
            train_loss, test_error = model.train(save_model=False)
            context['message'] = f"Train loss: {train_loss}, Test error: {test_error}"
            return render(request, "apnea_detection/train_test.html", context=context) 

    else:
        return render(request, "apnea_detection/train_test.html", context=context) 
 
#####################################################################
#                     Results
####################################################################

@login_required
def results(request):
    context = {}
    fieldnames = ['time','dataset','apnea_type','excerpt','sample_rate','scale_factor', \
                      'file','test_error','n_train','n_test','epochs']
    # fieldnames = [  'dataset',      'apnea_type',  'excerpt', 'epochs', 'batch_size', 'num_pos_train',    'num_neg_train',\
    #                 'num_pos_test', 'num_neg_test', 'precision_1',  'precision_0',  'recall_1', 'recall_0',  'f1_1', 'f1_0',\
    #                 'true_pos','true_neg','false_pos','false_neg' ]

    # render results csv as html
    results_file = f"{RESULTS_DIR}/results.csv"
    context["results"] = csv_to_html(results_file)
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