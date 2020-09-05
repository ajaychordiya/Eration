# Generated by Django 2.2.13 on 2020-08-12 09:30

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ration', '0013_auto_20200510_1642'),
    ]

    operations = [
        migrations.AddField(
            model_name='distgrains',
            name='dal',
            field=models.IntegerField(default=1, max_length=2),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='distgrains',
            name='guid',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='ration.family'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='distgrains',
            name='id3',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='distgrains',
            name='rice',
            field=models.IntegerField(default=1, max_length=2),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='distgrains',
            name='wheat',
            field=models.IntegerField(default=1, max_length=2),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='family',
            name='aadhar',
            field=models.TextField(default=1, max_length=12),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='newmem_reg',
            name='aadhar',
            field=models.TextField(default=1, max_length=12),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='registration',
            name='aadhar',
            field=models.CharField(default=1, max_length=20),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='family',
            name='mobileno',
            field=models.TextField(max_length=11),
        ),
        migrations.AlterField(
            model_name='newmem_reg',
            name='mobileno',
            field=models.TextField(max_length=11),
        ),
        migrations.AlterField(
            model_name='registration',
            name='mobile',
            field=models.CharField(max_length=11),
        ),
    ]
