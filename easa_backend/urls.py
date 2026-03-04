from django.urls import path, include

urlpatterns = [
    path("api/", include("rag_api.urls")),
]