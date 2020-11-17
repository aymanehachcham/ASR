
import json
import nemo
import nemo_asr
from ruamel.yaml import YAML
from dl_colab_notebooks.audio import record_audio

WORK_DIR = 'ASR_Models/quartznet15x5_multidataset/'
MODEL_YAML = 'ASR_Models/quartznet15x5_multidataset/quartznet15x5.yaml'
CHECKPOINT_ENCODER = 'ASR_Models/quartznet15x5_multidataset/JasperEncoder-STEP-243800.pt'
CHECKPOINT_DECODER = 'ASR_Models/quartznet15x5_multidataset/JasperDecoderForCTC-STEP-243800.pt'

ENABLE_GRAM = False


class Nemo_ASR_Systems:
    def __init__(self):
        # Defining initial Sample Rate:
        self.SAMPLE_RATE = 16000

        # Read the yaml version of the model:
        yaml = YAML(typ="safe")
        with open(MODEL_YAML) as f:
            jasper_model_definition = yaml.load(f)
        self.labels = jasper_model_definition['labels']

        # Instantiate necessary Neural Modules
        self.neural_factory = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.CPU,
            backend=nemo.core.Backend.PyTorch)

        # Data preprocessing
        self.data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(factory=self.neural_factory)

        # Jasper Encoder
        self.jasper_encoder = nemo_asr.JasperEncoder(
            jasper=jasper_model_definition['JasperEncoder']['jasper'],
            activation=jasper_model_definition['JasperEncoder']['activation'],
            feat_in=jasper_model_definition['AudioToMelSpectrogramPreprocessor']['features'])
        self.jasper_encoder.restore_from(CHECKPOINT_ENCODER, local_rank=0)

        # Jasper Decoder
        self.jasper_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(self.labels))
        self.jasper_decoder.restore_from(CHECKPOINT_DECODER, local_rank=0)

        # Greedy mode enabled:
        self.greedy_decoder = nemo_asr.GreedyCTCDecoder()

    def run_inference(self, manifest, greedy=True):
        # Instantiate necessary neural modules
        data_layer = nemo_asr.AudioToTextDataLayer(
            shuffle=False,
            manifest_filepath=manifest,
            labels=self.labels, batch_size=1)

        # Define inference DAG
        audio_signal, audio_signal_len, _, _ = data_layer()
        processed_signal, processed_signal_len = self.data_preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len)

        encoded, encoded_len = self.jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)

        log_probs = self.jasper_decoder(encoder_output=encoded)
        predictions = self.greedy_decoder(log_probs=log_probs)

        if greedy:
            eval_tensors = [predictions]

        tensors = self.neural_factory.infer(tensors=eval_tensors)
        if greedy:
            from nemo_asr.helpers import post_process_predictions
            prediction = post_process_predictions(tensors[0], self.labels)
        else:
            prediction = tensors[0][0][0][0][1]

        return prediction

    def create_output_manifest(self, file_path):
        # create manifest
        manifest = dict()
        manifest['audio_filepath'] = file_path
        manifest['duration'] = 18000
        manifest['text'] = 'todo'

        with open(file_path + ".json", 'w') as fout:
            fout.write(json.dumps(manifest))

        return file_path + ".json"

    def record_audio(self, duration=10):
        audio = record_audio(duration, sample_rate=self.SAMPLE_RATE)
        return audio