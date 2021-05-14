from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
from django.forms.fields import IntegerField


DATABASE_CHOICES = models.CharField()
# Create your models here.
# User profile 
class Setup(models.Model):  
    DATASETS = [
        ("DREAMS", "dreams"),
        ("UCDDB", "uccdb"),
        ("MIT-BIH", "mit-bih")
    ]
    APNEA_TYPES = [
        ("OSA", "obstructive sleep apnea"),
        ("OSAHS", "hypopnea")
    ]
    dataset = models.CharField(max_length = 20,
                                choices = DATASETS,
                                default = "DREAMS")
    apnea_type = models.CharField(max_length = 20,
                                choices = APNEA_TYPES,
                                default = "OSA")
    excerpt = models.PositiveIntegerField(default=1)

    
    def __str__(self):  
        return f"{self.excerpt} from {self.database}, apnea type: {self.apnea_type}"

# Create your models here.
# User profile 
class Normalization(models.Model): 
    NORMALIZATION_TYPES = [
        ("Linear", "linear"),
        ("Nonlinear", "nonlinear")
    ]
    norm = models.CharField(max_length = 20,
                                choices = NORMALIZATION_TYPES,
                                default = "Linear") 
    slope_threshold= models.PositiveIntegerField(default=1)
    scale_factor_low = models.PositiveIntegerField(default=1)
    scale_factor_high = models.PositiveIntegerField(default=1) 
    def __str__(self):  
        return f"Normalization factor: {self.norm}"