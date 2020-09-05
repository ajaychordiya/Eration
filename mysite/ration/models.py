
from django.db import models

# Create your models here.
class Registration(models.Model):
    name = models.CharField(max_length=100)
    dob = models.CharField(max_length=20)
    gender = models.CharField(max_length=10)
    aadhar = models.CharField(max_length=20)
    mobile = models.CharField(max_length=11)

class family(models.Model):
    first_name = models.CharField(max_length=10)
    last_name = models.CharField(max_length=10)
    dob = models.DateField()
    gender = models.CharField(max_length=6)
    mobileno = models.TextField(max_length=11)
    aadhar = models.TextField(max_length=12)
    card_color= models.CharField(max_length=10)
    def __str__(self):
        return self.id

class Newmem_reg(models.Model):
    first_name = models.CharField(max_length=10)
    last_name = models.CharField(max_length=10)
    dob = models.DateField()
    gender = models.CharField(max_length=6)
    mobileno = models.TextField(max_length=11)
    aadhar = models.TextField(max_length=12)
    uid = models.ForeignKey(family, on_delete=models.CASCADE)

    def __str__(self):
        return self.first_name

class distgrains(models.Model):
    wheat = models.IntegerField(max_length=2)
    rice = models.IntegerField(max_length=2)
    dal = models.IntegerField(max_length=2)
    dt = models.DateField()
    id3 = models.IntegerField()
    guid = models.ForeignKey(family, on_delete=models.CASCADE)

class gtable(models.Model):
    avail: int