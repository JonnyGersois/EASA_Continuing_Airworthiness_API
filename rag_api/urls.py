from django.urls import path
from .views import query_easa

urlpatterns = [
    path("query/", query_easa),
]