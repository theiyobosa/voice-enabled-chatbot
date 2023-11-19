# Voice Enabled Chatbot

This is a program that allows a user to use voice recognition to interact with a chatbot in English.
The chatbot also responds using speech. All of the models used here were Open Source models.

# Tools

- LLM: [LLaMA 2](https://ai.meta.com/llama/)
- Speech to Text Model: [OpenAI Whisper](https://github.com/openai/whisper)
- Text to Speech Model: [Coqui AI](https://github.com/coqui-ai/TTS/tree/dev)

# Setup

Clone the repository.

```
git clone https://github.com/theiyobosa/voice-enabled-chatbot.git
```

Create a virtual environment, and activate it.

```
python3 -m venv venv
source venv/bin/activate
```

Install the Python requirements.

```
pip install -r requirements.txt
```

Download and setup the LLaMa chat model of your choice using the instructions described [here](https://medium.com/@karankakwani/build-and-run-llama2-llm-locally-a3b393c1570e). You could use anyone of the LLaMA 2 7B-chat, 13B-chat, 70B-chat models. If you have a computer with limited RAM, it's best to use the quantized models, since they are lesser in size.

Run the program by passing in the `--model-path` argument, that indicates the path of the (.bin) LLaMA 2 model.

```
python3 main.py --model-path <model path>
```

There are other arguments that can be passed in, you can get more details on the other parameters by running:

```
python3 main.py -h
```

Depending on your computer, you may need administrative access to run the program. This will only be neccessary if you run into permision errors.

```
sudo python3 main.py --model-path <model path>
```