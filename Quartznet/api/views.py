from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.views.decorators.cache import never_cache

from .models import Sound
from ASR_Models.initialize_models import Nemo_ASR_Systems

import shutil
import os
import uuid
import sounddevice as sd
from scipy.io.wavfile import write



@api_view(['GET'])
@never_cache
def test_api(request):
    return Response({'response':"You are successfully connected to the ASR API"})


@api_view(['POST'])
def asr_conversion(request):
    asr_model = Nemo_ASR_Systems()
    data = request.FILES['audio']
    path = "Audio/" + data.name
    audio = Sound.objects.create(audio_file=data)
    json_manifest = asr_model.create_output_manifest(audio.audio_file.path)

    transcription = asr_model.run_inference(json_manifest)

    return Response({'Output': transcription})

@api_view(['POST'])
def post_record(request):
    asr_model = Nemo_ASR_Systems()
    fs = asr_model.SAMPLE_RATE  # Sample rate
    seconds = 5  # Duration of recording

    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    num_generated = uuid.uuid1()
    path = "Audio/sound_recorded_%02d.wav" % num_generated
    write(path, fs, recording)  # Save as WAV file
    audio = Sound.objects.create(audio_file=path)
    json_manifest = asr_model.create_output_manifest(audio.audio_file.path)

    return Response({'Output': json_manifest})

@api_view(['DELETE'])
def delete(request):
    folder_input = 'Audio/'
    for filename in os.listdir(folder_input):
        file_path = os.path.join(folder_input, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return Response({'response': "Audio files cleaned up!!"})