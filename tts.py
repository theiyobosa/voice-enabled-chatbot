from typing import Literal
import re
import sys
from playsound import playsound
from TTS.api import TTS


class HiddenPrints:
    # Hide print statements be rerouting the standard output.
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class TextToSpeech:
    """
    TTS wrapper for converting text to speech. The Open Source model being used
    here is a model from coqui-ai.

    -> params
    @tts_model: The name of the TTS model to use. This class uses coqui-ai's
        TTS models. You can find the list of all the TTS model names with:
        ```
        from TTS.api import TTS
        print(TTS().list_models())
        ```
    @device: The device to run the model on.
    @audio_filepath: The filepath of the audio file that the converted text will
        be saved to.
    @verbose: Print verbose output.
    """

    def __init__(self,
                 tts_model: str='tts_models/en/ljspeech/tacotron2-DCA',
                 device: Literal['cpu', 'cuda', 'mps']='cpu',
                 audio_filepath: str='audio.wav',
                 verbose: bool=False):
        if verbose:
            self.tts = TTS(model_name=tts_model,
                           progress_bar=True).to(device)
        else:
            with HiddenPrints():
                self.tts = TTS(model_name=tts_model, 
                               progress_bar=False).to(device)
        
        self.verbose = verbose
        self.audio_filepath = audio_filepath
        
    def to_speech(self, text: str):
        """
        Convert a piece of text to audio, and read it aloud.

        -> params
        @text: Piece of text to be converted to string and read aloud.
        """
        text = re.sub(r'\*.*?\*', '', text)

        if self.verbose:
            self.tts.tts_to_file(text=text, file_path=self.audio_filepath)
        else:
            with HiddenPrints():
                self.tts.tts_to_file(text=text, file_path=self.audio_filepath)
                
        playsound(self.audio_filepath)