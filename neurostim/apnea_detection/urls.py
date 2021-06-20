from django.urls import path
from apnea_detection import views 

urlpatterns =[ 
    # login/logout 
    path('login/', views.login_action, name="login"),
    path('register/', views.register_action, name="register"),
    path('logout/', views.logout_action, name="logout"),
    # home
    path('home', views.home, name="home"),

    # pipeline
    path('preprocessing/', views.preprocessing, name="preprocessing"),
    path('inference/', views.inference, name="inference"),
    path('results/', views.results, name="results")
]