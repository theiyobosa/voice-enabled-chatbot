from typing import Generator, Optional
from llama_cpp import ChatCompletionMessage, Llama

class LLM:
    """
    Handles answering to user's query with LLaMA model.

    -> params
    @model_path: That path of the saved LLaMa chat model.
    @context_size: Context size of the model.
    @verbose: Print verbose output.
    """

    def __init__(self,
                 model_path: str,
                 context_size: Optional[int]=1024,
                 verbose: Optional[bool]=False):
        
        self.model = Llama(model_path=model_path,
                           n_ctx=context_size,
                           verbose=verbose,
                           n_gpu_layers=1,
                           use_mlock=True)
        
        self.history = []
        self.last_response = ''


    def streamed_response(self,
                          query: str,
                          n: int,
                          max_tokens: int,
                          temperature: float) -> Generator[str, None, None]:
        """
        Prompt the LLM for a response, the output will be streamed.

        -> params
        @query: The user's query.
        @n: Number of conversations being sent to to LLaMA, `query` included.
        @max_tokens: The maximum munber of tokens to return in the response.
        @temperature: The temperature of the model.

        -> returns
        @response: The response of the LLM.
        """
        user_message = ChatCompletionMessage(role='user',
                                             content=query)
        self.history.append(user_message)
        
        convo_history = self.history[-n:]
        response = ''
        for op in self.model.create_chat_completion(convo_history,
                                                    max_tokens=max_tokens,
                                                    temperature=temperature,
                                                    stream=True):
            text = op['choices'][0]['delta'].get('content')
            if text:
                response += text
                yield text

        ai_message = ChatCompletionMessage(role='assistant',
                                           content=response)
        self.history.append(ai_message)
        self.last_response = response


    def unstreamed_response(self,
                            query: str,
                            n: int,
                            max_tokens: int,
                            temperature: float) -> str:
        """
        Prompt the LLM for a response, the output will not be streamed.

        -> params
        @query: The user's query.
        @n: Number of conversations being sent to to LLaMA, `query` included.
        @max_tokens: The maximum munber of tokens to return in the response.
        @temperature: The temperature of the model.

        -> returns
        @response: The response of the LLM.
        """
        user_message = ChatCompletionMessage(role='user',
                                             content=query)
        self.history.append(user_message)
        
        convo_history = self.history[-n:]
        output = self.model.create_chat_completion(convo_history,
                                                   max_tokens=max_tokens,
                                                   temperature=temperature,
                                                   stream=False)
        text = output['choices'][0]['message']
        ai_message = ChatCompletionMessage(role='assistant',
                                           content=text)
        self.history.append(ai_message)
        self.last_response = text
        return text