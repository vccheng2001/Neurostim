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
from apnea_detection.forms import LoginForm, RegisterForm, SetupForm, ModelHyperParamsForm
from apnea_detection.models import Setup, ModelHyperParams

import pandas as pd
import os
import csv
from datetime import date, datetime
from subprocess import Popen, PIPE, STDOUT
from sklearn import linear_model 
from pathlib import Path

# disable debugging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# directories 
ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = os.path.join(ROOT_DIR, "data")
INFO_DIR = os.path.join(ROOT_DIR, "info")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")

''' helper function to convert csv to html '''
def csv_to_html(file):
    df = pd.read_csv(file)
    html = df.to_html(classes='table table-striped table-bordered table-responsive table-sm')
    return html


@login_required
def home(request):
    context = {}
    return render(request, "apnea_detection/home.html", context=context)

@login_required
def setup(request):
    context = {}
    logs_file = f"{INFO_DIR}/log.csv"
    if request.method == "POST":
        form = SetupForm(request.POST)
        if form.is_valid():
            try:
                # normalize file and save form
                normalized_file, flatline_detection_params = normalize(form.cleaned_data)
                run_flatline_detection(flatline_detection_params)
                form.save()
                # display success message
                context["message"] = f"Successfully saved normalized file to {normalized_file}."
                # return new form 
                context['form'] =  SetupForm()
                # logs file
                context['logs'] = csv_to_html(logs_file)
                return render(request, "apnea_detection/setup.html", context=context)
            except Exception as error_message:
                # else throw error 
                context["error_heading"] = "Error during normalization step. Please try again."
                context["error_message"] = error_message
                return render(request, "apnea_detection/error.html", context=context)
    # if GET request
    context = {'form': SetupForm(), 'logs': csv_to_html(logs_file)} 
    return render(request, "apnea_detection/setup.html", context=context)

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
        fieldnames = ['time','DB','patient','samplingRate','action','status','file_folder_Name','parameters']
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


def run_flatline_detection(flatline_detection_params):
    dataset, apnea_type, excerpt, sample_rate, scale_factor = flatline_detection_params
   

    cmd = ["python", "flatline_detection.py", 
            "-d",  str(dataset), 
            "-a",  str(apnea_type),
            "-ex", str(excerpt),
            "-sr", str(sample_rate),
            "-sc", str(scale_factor)]


    proc = Popen(cmd, universal_newlines=True, 
                      stdout=PIPE, stderr=PIPE)

    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr

@login_required
def inference(request):

    context = {}
    results_file = f"{INFO_DIR}/summary_results.csv"

    # most recent setup parameters
    setup_params = Setup.objects.last()


    try:
        returncode, stdout, stderr = run(setup_params, None, test=True)
        results = get_summary_results(results_file)
        # save model hyperparameters
        # display success message
        context["message"] = f"Successfully performed inference."
        context["results"] = results
        context["setup_params"] = setup_params
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




''' train/testmodel using specified model hyperparameters '''
def run(setup_params, model_params, test):


    apnea_type = setup_params.apnea_type
    dataset = setup_params.dataset
    excerpt = setup_params.excerpt
    if model_params:
        epochs = model_params["epochs"]
        batch_size = model_params["batch_size"]
        threshold = model_params["positive_threshold"]
    # run apnea detection script

    if test:
        cmd = ["python", "train.py", 
               "-d", dataset, 
               "-a", apnea_type,
               "-ex", str(excerpt),
               "-t", "120",
               "-th", str(0.7),
               "--test"]
    else:
        cmd = ["python", "train.py", \
                        "-d", dataset,
                        "-a", apnea_type,
                        "-ex", str(excerpt),
                        "-t", "120",
                        "-ep", str(epochs),
                        "-b", str(batch_size)]

    proc = Popen(cmd, universal_newlines=True, 
                      stdout=PIPE, stderr=PIPE)

    stdout, stderr = proc.communicate()
    print('stdout of run', stdout, stderr)
    return proc.returncode, stdout, stderr 



@login_required
def train(request):
    context = {}
    results_file = f"{INFO_DIR}/summary_results.csv"

    # most recent setup parameters
    # choose from setup params 
    setup_params = Setup.objects.last()

    if request.method == "POST":
        form = ModelHyperParamsForm(request.POST)
        if form.is_valid():
            try:
                model_params = form.cleaned_data
                returncode, stdout, stderr = run(setup_params, model_params, False)

                saved_model_path = stdout
                # display success message
                context["message"] = f"Successfully saved trained model to {saved_model_path}"
                context["setup_params"] = setup_params
                context["model_params"] = model_params
              
                return render(request, "apnea_detection/inference.html", context=context)
            
            
            except Exception as error_message:
                # else throw error 
                context["error_heading"] = "Error during training. Please try again."
                context["error_message"] = error_message
                return render(request, "apnea_detection/error.html", context=context)
    # if GET request
    context = {'form': ModelHyperParamsForm()}
    return render(request, "apnea_detection/train.html", context=context)




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