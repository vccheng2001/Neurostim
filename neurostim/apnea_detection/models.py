from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.forms.fields import IntegerField


DATASETS = [
        ("dreams", "DREAMS Apnea database"),
        ("dublin", "University College of Dublin Database"),
        ("mit", "MIT-BIH Arrhythmia Database"),
        ("patch", "Neurostim Patch data")
]
APNEA_TYPES = [
    ("osa", "Obstructive sleep apnea"),
    ("osahs", "Hypopnea")
]
NORMALIZATION_TYPES = [
        ("linear", "linear scaling"),
        ("nonlinear", "nonlinear scaling")
]
# Create your models here.
# User profile 
class Preprocessing(models.Model):  
    dataset = models.CharField(max_length = 20,
                                choices = DATASETS,
                                default = "DREAMS")
    apnea_type = models.CharField(max_length = 20,
                                choices = APNEA_TYPES,
                                default = "OSA")
    excerpt = models.PositiveIntegerField(default=1)
    norm = models.CharField(max_length = 20,
                                choices = NORMALIZATION_TYPES,
                                default = "Linear") 
    slope_threshold= models.FloatField(default=0.025)
    scale_factor_low = models.PositiveIntegerField(default=1)
    scale_factor_high = models.PositiveIntegerField(default=100) 
    sample_rate = models.PositiveIntegerField(default=8)  


# Model hyperparameters
class ModelHyperParams(models.Model):  
    batch_size = models.PositiveIntegerField(default=32)
    epochs = models.PositiveIntegerField(default=10)
    positive_threshold = models.FloatField(default=0.7)

