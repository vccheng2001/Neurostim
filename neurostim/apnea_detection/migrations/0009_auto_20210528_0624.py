# Generated by Django 3.1.6 on 2021-05-28 06:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apnea_detection', '0008_modelparams'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='ModelParams',
            new_name='ModelHyperParams',
        ),
        migrations.AlterField(
            model_name='setup',
            name='scale_factor_high',
            field=models.PositiveIntegerField(default=100),
        ),
        migrations.AlterField(
            model_name='setup',
            name='slope_threshold',
            field=models.FloatField(default=0.025),
        ),
    ]