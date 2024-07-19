import subprocess 
import json
import re
import pandas as pd

# Input path for the json file
input_path = 'resources/input_to_rnn_training.json'

# Trace Data path
output_path = 'trace_data/max_alpha_df.csv'

# Create a df 
trace_df = pd.DataFrame(columns =['reg_method','weight', 'loss','forget_rate' ,'xQf', 'alpha', 'xQr','correlated_update', 'perseveration_bias', 'xH', 'beta', 'sindy_beta'])

# Regularization method. change this if you change the reg method in rnn_training
reg_method = '2nd_approach'

# Initial data
input_to_rnn_training = {
    "weight_reg_rnn": 0.015264
}

# Save input_to_rnn as json
with open(input_path, 'w') as json_file:
    json.dump(input_to_rnn_training, json_file, indent=4)

# Function to update the weight_reg_rnn
def update_weight_reg_rnn(new_weight):
    input_to_rnn_training['weight_reg_rnn'] = new_weight
    with open(input_path, 'w') as json_file:
        json.dump(input_to_rnn_training, json_file, indent=4)



try:
    for i in range(20):
        # Opening the json file 
        with open('resources/input_to_rnn_training.json') as file:
            trace_input = json.load(file)
        # Tracking the value in the json file
        weight_reg_rnn = trace_input['weight_reg_rnn']

        # Run rnn_main and save the output 
        output_rnn = subprocess.run(['python', 'rnn_main.py'], capture_output=True, text=True)
        output_rnn_string = output_rnn.stdout
        print(output_rnn_string)

        # Run sindy_main and save the output
        output_sindy = subprocess.run(['python', 'sindy_main.py'], capture_output=True, text=True)
        output_sindy_string = output_sindy.stdout
        print(output_sindy_string)

        #####################################################
        # Getting all relevant informations from the output #
        #####################################################

        # loss value of the rnn
        loss_value = re.search(r'Epoch 1/1 --- Loss: ([\d\.]+);', output_rnn_string).group(1)
        loss_value = float(loss_value)

        # forget_rate of the ground truth
        forget_rate = re.search(r'forget_rate\s*=\s*([\d\.]+)', output_rnn_string)  
        forget_rate = float(forget_rate.group(1))

        # xQf of sindy
        xQf = re.search(r'\(xQf\)\[k\+1\] = ([^\n]+)', output_sindy_string)
        xQf = xQf.group(1).strip()

        # alpha
        alpha = re.search(r'alpha\s*=\s*([\d\.]+)', output_rnn_string)  
        alpha = float(alpha.group(1))

        # xQr of sindy
        xQr = re.search(r'\(xQr\)\[k\+1\] = ([^\n]+)', output_sindy_string)
        xQr = xQr.group(1).strip()

        # correlated_update
        correlated_update = re.search(r'correlated_update\s*=\s*(True|False)', output_rnn_string)  
        correlated_update = correlated_update.group(1) == 'True'

        # xQc of sindy
        # xQc = re.search(r'\(xQc\)\[k\+1\] = ([^\n]+)', output_sindy_string)
        # xQc = xQc.group(1).strip()    

        # perseveration_bias
        perseveration_bias = re.search(r'perseveration_bias\s*=\s*([\d\.]+)', output_rnn_string)  
        perseveration_bias = float(perseveration_bias.group(1))   

        # xH of sindy
        xH = re.search(r'\(xH\)\[k\+1\] = ([^\n]+)', output_sindy_string)
        xH = xH.group(1).strip()    

        # beta
        beta = re.search(r'beta\s*=\s*([\d\.]+)', output_rnn_string)  
        beta = float(beta.group(1))

        # sindy_beta
        sindy_beta = re.search(r'Beta for SINDy:\s*([\d\.]+)', output_sindy_string)  
        sindy_beta = float(sindy_beta.group(1))

        # add all values 
        trace_df.loc[len(trace_df.index)] = [reg_method, weight_reg_rnn, loss_value, forget_rate, xQf, alpha, xQr, correlated_update, perseveration_bias, xH, beta, sindy_beta]

except KeyboardInterrupt:
    pass

print(trace_df)

trace_df.to_csv(output_path)