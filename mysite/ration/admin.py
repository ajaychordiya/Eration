from django.contrib import admin
from ration.models import family,Newmem_reg,distgrains
    #,distgrains
# Register your models here.

admin.site.register(family)
admin.site.register(Newmem_reg)

admin.site.register(distgrains)