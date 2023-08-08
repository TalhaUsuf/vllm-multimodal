from vllm import LLM, SamplingParams
from transformers import LlamaTokenizerFast

prompts = [
    "A photo of",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llama_tok = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenized = llama_tok(prompts, return_tensors='pt')

llm = LLM(model="meta-llama/Llama-2-7b-hf")

input_embeddings = llm.get_input_embeddings()(tokenized.input_ids.to('cuda')) # shape [1, 4, 4096]

print(input_embeddings.shape)

print(llm.generate(prompts, sampling_params=sampling_params))
