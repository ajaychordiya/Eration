# Generated by Django 3.0.3 on 2020-03-24 07:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ration', '0004_delete_create_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='Registration',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('dob', models.DateField()),
                ('gender', models.CharField(max_length=10)),
                ('Aadhar', models.IntegerField()),
                ('mobile', models.IntegerField()),
            ],
        ),
    ]
