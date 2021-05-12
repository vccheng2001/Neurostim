from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls import include, url
from django.contrib import admin

from apnea_detection import views 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.login_action, name='login'),
    path('apnea_detection/', include('apnea_detection.urls'), name="apnea_detection"),

]
