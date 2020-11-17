from django.urls import include, path
from django.conf.urls.static import static
from . import views
from django.conf import settings


app_name = 'api'

urlpatterns = [
    path(r'test', views.test_api, name='test_api_communication'),
    path(r'asr/', views.asr_conversion),
    path(r'record/', views.post_record),
    path(r'del', views.delete)


] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)