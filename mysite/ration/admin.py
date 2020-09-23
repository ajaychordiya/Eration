from django.contrib import admin
from ration.models import family,mem_reg,distgrains
    #,distgrains
# Register your models here.

admin.site.register(family)
admin.site.register(mem_reg)

admin.site.register(distgrains)