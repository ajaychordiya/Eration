from django.urls import path
from . import views
urlpatterns =[
    path('', views.index, name='index'),
    path('Login', views.Login, name='Login'),
    path('Login2', views.Login2, name='Login2'),
    path('dist2', views.Login2, name='dist2'),
    path('Newmem_reg', views.Newmem_reg, name='Newmem_reg'),
    path('Newmem_page',views.Newmem_page,name='Newmem_page'),
    path('newfamily', views.newfamily, name='newfamily'),
    path('family_member/<int:id>', views.family_member, name='newfamily_member'),
    path('capture',views.capture, name='capt'),
    #path('addgrains',views.addgrains,name='addgrains'),
    path('distgrain',views.distgrain,name='distgrain'),
    #path('gdata',views.gdata,name='gdata'),
    path('recognize',views.recognize,name='recognize'),
    path('updateView',views.updateView,name='updateView'),
    path('deleteView',views.deleteView,name='deleteView'),
    path('saveUpdate/<int:id>',views.saveUpdate,name='saveUpdate'),
    path('dist',views.dist,name='dist')

]