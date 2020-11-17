from django.db import models

# Create your models here.
class Sound(models.Model):
    audio_file = models.FileField(upload_to='Audio/', null=True, blank=True)