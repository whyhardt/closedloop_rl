import json

with open('resources/input_to_rnn_training.json') as file:
    trace_input = json.load(file)

weight_reg_rnn: float = trace_input['weight_reg_rnn']

print(weight_reg_rnn)

