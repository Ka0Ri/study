from vllm.engine.llm_engine import LLMEngine
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter
from vllm.outputs import RequestOutput
from vllm import SamplingParams
from typing import List, Optional
import gradio as gr

class StreamingLLM:
    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        **kwargs,
    ) -> None:
        engine_args = EngineArgs(model=model, quantization=quantization, dtype=dtype, enforce_eager=True)
        self.llm_engine = LLMEngine.from_engine_args(engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

    def generate(
        self,
        prompt: Optional[str] = None,
        sampling_params: Optional[SamplingParams] = None
    ) -> List[RequestOutput]:
        
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, prompt, sampling_params)
        
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                yield output


llm = StreamingLLM(model="casperhansen/llama-3-8b-instruct-awq", quantization="AWQ", dtype="float16")
tokenizer = llm.llm_engine.tokenizer.tokenizer
sampling_params = SamplingParams(temperature=0.6,
                                 top_p=0.9,
                                 max_tokens=4096,
                                 stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                                )

def predict(message, history):
    history_chat_format = []
    for human, assistant in history:
        history_chat_format.append({"role": "user", "content": human })
        history_chat_format.append({"role": "assistant", "content": assistant})
    history_chat_format.append({"role": "user", "content": message})
  
    prompt = tokenizer.apply_chat_template(history_chat_format, tokenize=False)

    for chunk in llm.generate(prompt, sampling_params):
        yield chunk.outputs[0].text

gr.ChatInterface(predict).launch()