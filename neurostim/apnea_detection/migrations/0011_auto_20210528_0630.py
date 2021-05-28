# Generated by Django 3.1.6 on 2021-05-28 06:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apnea_detection', '0010_setup_sample_rate'),
    ]

    operations = [
        migrations.AlterField(
            model_name='setup',
            name='dataset',
            field=models.CharField(choices=[('dreams', 'DREAMS Apnea database'), ('dublin', 'University College of Dublin Database'), ('mit', 'MIT-BIH Arrhythmia Database')], default='DREAMS', max_length=20),
        ),
    ]