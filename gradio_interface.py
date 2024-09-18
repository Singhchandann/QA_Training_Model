import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import gradio as gr

def answer_question(question, model, tokenizer, max_len=512):
    inputs = tokenizer(question, return_tensors="pt", max_length=max_len, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load the pre-trained model and tokenizer
model_name = "facebook/bart-large"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define Gradio interface
interface = gr.Interface(
    fn=lambda question: answer_question(question, model, tokenizer),
    inputs="text",
    outputs="text",
    title="Financial Question Answering System"
)

# Launch the interface
interface.launch()
