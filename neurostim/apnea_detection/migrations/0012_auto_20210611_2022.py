# Generated by Django 3.1.6 on 2021-06-11 20:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('apnea_detection', '0011_auto_20210528_0630'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Setup',
            new_name='Preprocessing',
        ),
    ]