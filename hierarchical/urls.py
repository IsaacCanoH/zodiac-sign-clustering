from django.urls import path
from . import views

app_name = 'hierarchical'

urlpatterns = [
    path('workspace/', views.workspace, name='workspace'),
    path('train/', views.train, name='train'),
    path('activate/<int:run_id>/', views.activate, name='activate'),
    path('delete/<int:run_id>/', views.delete, name='delete'),
    path('results/<int:run_id>/', views.results, name='results'),
]