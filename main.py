from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify
import re

app = Flask(__name__)

model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, min_length=20,
                             do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    user_input = input('Você: ')
    if user_input.lower() == 'sair':
        break
    response = generate_response(user_input)
    print(f'Chatbot: {response}')


#Saving the model
model.save_pretrained('./chatbot_model')
tokenizer.save_pretrained('./chatbot_model')

#Loading the model
model = GPT2LMHeadModel.from_pretrained('./chatbot_model')
tokenizer = GPT2Tokenizer.from_pretrained('./chatbot_model')

#Preprocessing the text to avoid some errors
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

Max_Length = 100
@app.route('/chat', methods=['POST'])

def chat():
    user_input = request.json.get('text')
    if not user_input or len(user_input.strip()) == 0:
        return jsonify({'error': 'Input is empty'}), 400
    
    if len(user_input) > Max_Length:
        return jsonify({'error': f'Input too long. Max {Max_Length} characters'}), 400
    
    user_input = preprocess_text(user_input)
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)