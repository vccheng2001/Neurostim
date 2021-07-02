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
    path('visualize/', views.visualize, name="visualize"),
    path('flatline_detection/', views.flatline_detection, name="flatline_detection"),
    path('train_test/', views.train_test, name="train_test"),
    path('results/', views.results, name="results")
]