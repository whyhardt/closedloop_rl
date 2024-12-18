import sys, os

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__=='__main__':
    ls = [0, 1, 2, 3, 4]
    
    print(ls[0:-1])
    print(ls[1:None])