# Generated by Django 3.1.6 on 2021-05-21 00:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apnea_detection', '0006_auto_20210520_2322'),
    ]

    operations = [
        migrations.AlterField(
            model_name='setup',
            name='apnea_type',
            field=models.CharField(choices=[('osa', 'Obstructive sleep apnea'), ('osahs', 'Hypopnea')], default='OSA', max_length=20),
        ),
        migrations.AlterField(
            model_name='setup',
            name='dataset',
            field=models.CharField(choices=[('dreams', 'DREAMS Apnea database'), ('uccdb', 'University College of Dublin Database'), ('mit-bih', 'MIT-BIH Arrhythmia Database')], default='DREAMS', max_length=20),
        ),
        migrations.AlterField(
            model_name='setup',
            name='norm',
            field=models.CharField(choices=[('linear', 'linear scaling'), ('nonlinear', 'nonlinear scaling')], default='Linear', max_length=20),
        ),
    ]