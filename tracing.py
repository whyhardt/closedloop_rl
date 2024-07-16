import subprocess 
import json

# Initial data
input_to_rnn_training = {
    "weight_reg_rnn": 0.0
}

#Input path
input_path = 'resources/input_to_rnn_training.json'

#save 
with open(input_path, 'w') as json_file:
    json.dump(input_to_rnn_training, json_file, indent=4)

# Function to update the weight_reg_rnn
def update_weight_reg_rnn(new_weight):
    input_to_rnn_training['weight_reg_rnn'] = new_weight




#result = subprocess.run(['python', 'rnn_main.py'], capture_output=True, text=True)
#print("Output of script1.py:")
#print(result.stdout)