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


class UploadFile(models.Model):
    file = models.FileField()
    dataset = models.CharField(max_length = 20, blank=True)
    apnea_type = models.CharField(max_length = 20, blank=True)
    excerpt = models.PositiveIntegerField(default=1, blank=True)
    sample_rate = models.PositiveIntegerField(default=8, blank=True)  
    scale_factor = models.PositiveIntegerField(default=1, blank=True) 
