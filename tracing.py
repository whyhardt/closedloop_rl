import subprocess 
import json
import re
import pandas as pd

#Input path for the json file
input_path = 'resources/input_to_rnn_training.json'

# create a df 
trace_df = pd.DataFrame(columns =['Weight', 'Loss'])

# Initial data
input_to_rnn_training = {
    "weight_reg_rnn": 0.0
}

#save input_to_rnn as json
with open(input_path, 'w') as json_file:
    json.dump(input_to_rnn_training, json_file, indent=4)

# Function to update the weight_reg_rnn
def update_weight_reg_rnn(new_weight):
    input_to_rnn_training['weight_reg_rnn'] = new_weight
    with open(input_path, 'w') as json_file:
        json.dump(input_to_rnn_training, json_file, indent=4)

values = [1e-3, 1e-2, 1e-1, 1]

for value in values:
    with open('resources/input_to_rnn_training.json') as file:
        trace_input = json.load(file)

    weight_reg_rnn = trace_input['weight_reg_rnn']

    output_rnn = subprocess.run(['python', '/Users/sebastian/Library/Mobile Documents/com~apple~CloudDocs/Cognitive Science/24 SoSe/Lab Rotation RNN/closedloop_rl/rnn_main.py'], capture_output=True, text=True)
    output_rnn_string = output_rnn.stdout
    print(output_rnn_string)

    loss_value = re.findall(r'Epoch 1/1 --- Loss: ([\d\.]+);', output_rnn_string)
    loss_value = loss_value = float(loss_value[0])

    trace_df.loc[len(trace_df.index)] = [weight_reg_rnn, loss_value]

    new_weight = value
    update_weight_reg_rnn(new_weight)

print(trace_df)

output_sindy = subprocess.run(['python', '/Users/sebastian/Library/Mobile Documents/com~apple~CloudDocs/Cognitive Science/24 SoSe/Lab Rotation RNN/closedloop_rl/sindy_main.py'], capture_output=True, text=True)
output_sindy_string = output_sindy.stdout
print(output_sindy_string)