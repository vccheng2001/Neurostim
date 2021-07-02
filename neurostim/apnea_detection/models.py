from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.forms.fields import IntegerField
from decimal import Decimal

DATASETS = [
        ("dreams", "DREAMS Apnea database"),
        ("dublin", "University College of Dublin Database"),
        ("mit", "MIT-BIH Arrhythmia Database"),
        ("patchJeff", "Neurostim Patch data: Jeff"),
        ("patchAllister", "Neurostim Patch data: Allister")

]
APNEA_TYPES = [
    ("osa", "Obstructive sleep apnea"),
    ("osahs", "Hypopnea")
]
NORMALIZATION_TYPES = [
        ("linear", "linear scaling"),
        ("nonlinear", "nonlinear scaling")
]


class UploadFile(models.Model):
    file = models.FileField()
    dataset = models.CharField(max_length = 20, blank=True)
    apnea_type = models.CharField(max_length = 20, blank=True)
    excerpt = models.CharField(max_length = 20, blank=True)
    sample_rate = models.PositiveIntegerField(default=8, blank=True)  
    scale_factor = models.PositiveIntegerField(default=1, blank=True) 

class FlatlineDetectionParams(models.Model):
    flatline_thresh = models.FloatField(default=10, blank=True)
    low_thresh = models.DecimalField(max_digits=3, decimal_places=2, blank=True, default=0.1)
    high_thresh = models.DecimalField(max_digits=3, decimal_places=2, blank=True, default=0.5) 
