import subprocess 
import json

result = subprocess.run(['python', 'rnn_main.py'], capture_output=True, text=True)
print("Output of script1.py:")
print(result.stdout)