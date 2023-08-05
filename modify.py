from vllm import LLM, SamplingParams


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


llm = LLM(model="meta-llama/Llama-2-7b-hf")


print(llm.generate(prompts, sampling_params=sampling_params))
