"""
Validate functions of utils.py
"""

# register src directory path to PYTHONPATH
import sys
from os import path, pardir
current_dir = path.abspath(path.dirname(__file__))
parent_dir = path.abspath(path.join(current_dir, pardir))
parent_parent_dir = path.abspath(path.join(parent_dir, pardir))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd

def validate_vec_length_is_larger(in_len, lowwer_bound_len, OPT):
    """
    validate input length is larger than lowwer bound length
    hopely input length is larger than lowwer bound length

    :param  in_len           : int, input length
    :param  lowwer_bound_len : int, lowwer bound length
    :param  OPT              : '>' or '>='
    """

    if (OPT == '>'):
        assert (in_len > lowwer_bound_len), (
            'Input length is smaller than or equal to lowwer bound length.'
        )
    elif (OPT == '>='):
        assert (in_len >= lowwer_bound_len), (
            'Input length is smaller than lowwer bound length.'
        )
    else:
        print('Please set variable "OPT" correctly. OPT must be ">" or ">=".')

