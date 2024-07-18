import subprocess 
import json
import re
import pandas as pd

# Input path for the json file
input_path = 'resources/input_to_rnn_training.json'

# Trace Data path
output_path = 'trace_data/trace_df.csv'

# Create a df 
trace_df = pd.DataFrame(columns =['reg_method','weight', 'loss','forget_rate' ,'xQf', 'alpha', 'xQr','correlated_update', 'xQc', 'perseveration_bias', 'xH', 'beta', 'sindy_beta'])

# Regularization method. change this if you change the reg method in rnn_training
reg_method = '2nd_approach'

# Initial data
input_to_rnn_training = {
    "weight_reg_rnn": 0.0
}

# Save input_to_rnn as json
with open(input_path, 'w') as json_file:
    json.dump(input_to_rnn_training, json_file, indent=4)

# Function to update the weight_reg_rnn
def update_weight_reg_rnn(new_weight):
    input_to_rnn_training['weight_reg_rnn'] = new_weight
    with open(input_path, 'w') as json_file:
        json.dump(input_to_rnn_training, json_file, indent=4)

# All the weights you want tested
values = [
    1e-05,
    1.3257113655901082e-05,
    1.757510624854793e-05,
    2.3299518105153718e-05,
    3.0888435964774785e-05,
    4.094915062380427e-05,
    5.4286754393238594e-05,
    7.196856730011514e-05,
    9.540954763499944e-05,
    0.00012648552168552957,
    0.00016768329368110083,
    0.00022229964825261955,
    0.00029470517025518097,
    0.0003906939937054617,
    0.0005179474679231213,
    0.0006866488450042998,
    0.0009102981779915217,
    0.0012067926406393288,
    0.0015998587196060573,
    0.0021209508879201904,
    0.002811768697974231,
    0.003727593720314938,
    0.004941713361323833,
    0.006551285568595509,
    0.00868511373751352,
    0.01151395399326447,
    0.015264179671752334,
    0.020235896477251575,
    0.026826957952797246,
    0.03556480306223128,
    0.04714866363457394,
    0.0625055192527397,
    0.08286427728546843,
    0.10985411419875572,
    0.14563484775012445,
    0.19306977288832497,
    0.2559547922699533,
    0.33932217718953295,
    0.4498432668969444,
    0.5963623316594636,
    0.7906043210907702,
    1.0481131341546852,
    1.389495494373136,
    1.8420699693267164,
    2.44205309454865,
    3.2374575428176398,
    4.291934260128778,
    5.689866029018293,
    7.543120063354607,
    10.0
]

try:
    while True:
    # Loop through all the weights
        for value in values:
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
            xQc = re.search(r'\(xQc\)\[k\+1\] = ([^\n]+)', output_sindy_string)
            xQc = xQc.group(1).strip()    

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
            trace_df.loc[len(trace_df.index)] = [reg_method, weight_reg_rnn, loss_value, forget_rate, xQf, alpha, xQr, correlated_update, xQc, perseveration_bias, xH, beta, sindy_beta]

            # Update the weight in the json file 
            new_weight = value
            update_weight_reg_rnn(new_weight)
except KeyboardInterrupt:
    pass

print(trace_df)

trace_df.to_csv(output_path)