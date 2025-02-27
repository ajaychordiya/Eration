# Generated by Django 3.0.6 on 2020-05-10 11:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ration', '0012_auto_20200410_1736'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='distgrains',
            name='dal',
        ),
        migrations.RemoveField(
            model_name='distgrains',
            name='guid',
        ),
        migrations.RemoveField(
            model_name='distgrains',
            name='rice',
        ),
        migrations.RemoveField(
            model_name='distgrains',
            name='rokel',
        ),
        migrations.RemoveField(
            model_name='distgrains',
            name='wheat',
        ),
        migrations.RemoveField(
            model_name='family',
            name='aadhar',
        ),
        migrations.RemoveField(
            model_name='newmem_reg',
            name='aadhar',
        ),
        migrations.RemoveField(
            model_name='registration',
            name='aadhar',
        ),
        migrations.AddField(
            model_name='distgrains',
            name='dt',
            field=models.DateField(default='2020-05-10'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='family',
            name='card_color',
            field=models.CharField(default='Yellow', max_length=10),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='family',
            name='mobileno',
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name='newmem_reg',
            name='mobileno',
            field=models.CharField(max_length=10),
        ),
    ]
