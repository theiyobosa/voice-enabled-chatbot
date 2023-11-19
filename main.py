from argparse import ArgumentParser

from llm import LLM
from tts import TextToSpeech
from stt import SpeechToText

parser = ArgumentParser()

parser.add_argument(
    '-mp',
    '--model-path', 
    type=str,
    required=True,
    help="The path to the LLaMA 2 model.",
)

parser.add_argument(
    '-cs',
    '--context-size',
    type=int,
    required=False,
    default=2048,
    help='The context size, in number of tokens for the LLaMA 2 model.'
)

parser.add_argument(
    '-ms',
    '--max-size',
    type=int,
    required=False,
    default=1024,
    help="The maximum size of the number of tokens that should be returned in \
        LLaMA 2 model's output."
)

parser.add_argument(
    '-tp',
    '--temperature',
    type=float,
    required=False,
    default=0.2,
    help="The temperature parameter for the model."
)

parser.add_argument(
    '-td',
    '--tts-device',
    type=str,
    choices=['cpu', 'cuda', 'mps'],
    required=False,
    default='mps',
    help='The device to run the text to speech model on, could either be cpu, \
        cuda or mps.'
)

parser.add_argument(
    '-sd',
    '--stt-device',
    type=str,
    choices=['cpu', 'cuda'],
    required=False,
    default='cpu',
    help='The device to run the speech to text model on, could either be cpu or\
        cuda.'
)

parser.add_argument(
    '-sm',
    '--stt-model',
    type=str,
    choices=['tiny.en', 'base.en', 'small.en', 'medium.en'],
    required=False,
    default='small.en',
    help='The speech to text model to use.'
)

args = parser.parse_args()

tts = TextToSpeech(device=args.tts_device)

stt = SpeechToText(stt_model=args.stt_model,
                   device=args.stt_device)

llm = LLM(model_path=args.model_path, #"../LLMs/llama/llama-2-13b-chat/ggml-model-q4_0.bin",
          context_size=args.context_size)

audio_filename = 'output.csv'

while True:
    # Record user's voice query
    print('\n\n[Hold SPACEBAR to record]')
    stt.record(filepath=audio_filename)

    # Convert recorded audio to text
    print('[Processing recording]')
    query = stt.to_text(audio_filepath=audio_filename)

    print('USER: ', query)

    # Get model's reqponse from user's query
    print('\n\nAI: ', end='', flush=True)
    for t in llm.streamed_response(query=query,
                                   n=3,
                                   max_tokens=args.max_size,
                                   temperature=args.temperature):
        print(t, end='', flush=True)
    
    # Read out the model's response
    tts.to_speech(llm.last_response)