from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_llm_answer(context, question):
    prompt_template = """Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = prompt_template.format(context=context, question=question)

    # Truncate the prompt if it's too long
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if input_ids.shape[1] > 2048:
        max_context_tokens = 2048 - 200
        context_tokens = tokenizer(context, return_tensors="pt").input_ids[0][:max_context_tokens]
        context = tokenizer.decode(context_tokens, skip_special_tokens=True)
        prompt = prompt_template.format(context=context, question=question)

    result = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].split("Answer:")[-1].strip()
