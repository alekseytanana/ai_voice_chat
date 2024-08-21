from datetime import datetime
import openvino_genai as gen

from openvino_genai.py_generate_pipeline import GenerationConfig


pipe = gen.LLMPipeline(r'C:\Users\atanana\PycharmProjects\openvino.genai\TinyLlama-1.1B-Chat-v1.0')
config = GenerationConfig(max_new_tokens=20, num_beam_groups=3, num_beams=15, diversity_penalty=1.5)
pipe.set_generation_config(config)
pipe.start_chat()

while True:
    prompt = input('[Me]: ')
    if prompt == 'quit':
        break
    start = datetime.now()
    print(f'[Bot]: {pipe(prompt)}')
    finish = datetime.now()
    took = finish - start
    print(f'    (took {took.seconds} seconds)')

pipe.finish_chat()
