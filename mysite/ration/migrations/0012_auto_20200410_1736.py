# Generated by Django 3.0.3 on 2020-04-10 12:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ration', '0011_auto_20200410_1729'),
    ]

    operations = [
        migrations.AlterField(
            model_name='newmem_reg',
            name='aadhar',
            field=models.BigIntegerField(),
        ),
    ]
