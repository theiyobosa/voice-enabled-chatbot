from typing import Literal

import keyboard
import pyaudio
import wave
import whisper

class SpeechToText:
    """
    Wrapper class for converting Speech to Text. The model being used here is
    OpenAI's whisper model.

    -> params
    @stt_model: The name of the Whisper Speech to Text model to use.
    @device: The device to run the model on. Currently, M1 GPUs don't seem to be
        supported, so it's either just `'cpu'` or `'cuda'`.
    """

    def __init__(self, 
                 stt_model: str='base.en',
                 device: Literal['cpu', 'cuda']='cpu'):
        self.model = whisper.load_model(stt_model, 
                                        device=device)
        

    def record(self,
               chunk: int=1024,
               sample_format :int=pyaudio.paInt16,
               channels :int=1,
               rate :int=44100,
               filepath :str="output.wav"
    ):
        """
        Record audio while the space key on the keyboard is pressed.

        -> params
        @chunk: Chunk size for the recording.
        @sample_format: Sampling format.
        @channels: Number of channels captured when recording.
        @rate: Sampling rate.
        @filepath: The filepath where the recording will be saved.
        """
        
        p = pyaudio.PyAudio()

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=rate,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []

        keyboard.wait('space')
        if keyboard.is_pressed('space'):

            while keyboard.is_pressed('space'):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        print('[Finished recording]')

        wf = wave.open(filepath, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()


    def to_text(self, audio_filepath: str) -> str:
        """
        Convert audio file to text.

        -> params
        @audio_filepath: The file path of the audio file. 

        -> returns
        @output: The text from the audio.
        """
        output = self.model.transcribe(audio_filepath, fp16=False)
        output = output['text']
        return output