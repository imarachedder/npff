"""
URL configuration for pension_forecast project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from pension_forecast_app import views
from pension_forecast_app.views import DatasetPredictionView


schema_view = get_schema_view(
    openapi.Info(
        title="Your API",
        default_version='v1',
        description="Описание вашего API",
        contact=openapi.Contact(email="contact@example.com"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path("info", views.welcome_view, name="info"),
    path('admin/', admin.site.urls),
    # path('predict/', DatasetPredictionView.as_view(), name='predict'),
    path('predict/', DatasetPredictionView.as_view(), name='predict'),  # Для POST запроса
    # path('predict/results/', DatasetPredictionView.as_view(), name='predict-results'),
    # Для GET запросаpath('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),

]
# urlpatterns += router.urls
# urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)