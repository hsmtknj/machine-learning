# coding: utf-8

"""
General Utilities Module
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
import tensorflow as tf
import pickle
import os

from modeldev import d_layer
from modeldev import d_model
from mylib import utils
from mylib import validation_utils


def remove_col_containing_specific_str(in_df, specific_str):
    """
    remove columns including specific string from input data frame

    :param  in_df        : pandas.DataFrame, input data frame
    :param  specific_str : str, deletion columns from in_df
    :return out_df       : pandas.DataFrame, data frame that spcific columns is removed
    """

    # see if input data frame has specific string on columns or not
    flag_containing = in_df.columns.str.contains(specific_str)

    # remove columns if input data frame has it
    num_containing = sum(flag_containing)
    if (num_containing >= 1):
        out_df = in_df.drop(columns=in_df.columns[flag_containing])
    else:
        out_df = in_df
    return out_df


def convert_parts_of_str(in_str, target_str, replacing_str):
    """
    convert target string into replacing string of input string
    e.g.
        s_before = 'ab-cd_e'
        s_after = convert_parts_of_str(s_before, 'c', 'K')
        print(s_after)  # ab-Kd_e
    
    :param  in_str        : str, input string
    :param  target_str    : str, string which is subject to be replaced
    :param  replacing_str : str, replacing string
    :return out_str       : str, output string
    """

    # convert str object into list object (because str is immutable)
    strlist = np.array(list(in_str))

    # detect target string
    ind = np.where(strlist == target_str)
    if (len(ind[0]) == 0):
        out_str = in_str
        return out_str

    # replace string
    strlist[ind[0]] = replacing_str
    out_str = ''.join(strlist)
    return out_str


def detect_first_timing_from_events(in_vec, interval):
    """
    detect first event timing from series of events
    regard events which are within iterval threshold as same event group

    e.g. interval = 5
        input  : [5, 8, 9, 11, 31, 32, 35, 39, 101, 103]
        output : [0, 4, 8]

    :param  in_vec  : ndarray, events timing vector
    :param  iterval : int, interval threshold
    :return out_vec : ndarray, timing vector that first timing is detected from events
    """
    
    # check differences from previous timestamp
    one_slide_vec = np.zeros(len(in_vec))
    one_slide_vec[1:] = in_vec[0:len(in_vec)-1]
    diff_vec = abs(in_vec - one_slide_vec)

    # first index must be first timing of events
    diff_vec[0] = interval + 1

    # detect groups
    timestamp_diff_df = pd.DataFrame(diff_vec)
    timestamp_diff_df = timestamp_diff_df[timestamp_diff_df[0] > interval]
    out_vec = np.array(timestamp_diff_df.index)

    return out_vec


def convert_ind_to_boolind(ind_vec, bool_list_length):
    """
    convert index numbers list to oolian index list
    e.g.
        input  : [0, 1, 5]
        output : [True, True, False, False, False, True]

    :param  ind_vec          : ndarray, 
    :param  bool_list_length : int, 
    :return bool_vec         : ndarray, 
    """
    
    validation_utils.validate_vec_length_is_larger(bool_list_length, max(ind_vec), '>')

    if (len(ind_vec) == 0):
        return None
    else:
        out_vec = np.full(bool_list_length, False)
        out_vec[ind_vec] = True
        return out_vec
    

def load_dataset_X_y(dirname, opt):
    """
    load training data

    :param  : dirname : str, loading target directory
    :param  : opt     : str, option data format "pandas" or "numpy"
    :return : data_X  : numpy, training data
    :return : data_y  : numpy, true data
    """
    
    # X
    input_filename = dirname + '/data_X.csv'
    df_X = pd.read_csv(input_filename, encoding='shift-jis')

    # y
    input_filename = dirname + '/data_y.csv'
    df_y = pd.read_csv(input_filename, encoding='shift-jis')

    if (opt == 'pd'):
        return df_X, df_y
    elif (opt == 'np'):
        return df_X.values, df_y.values


def translate_contents_without_string(input):
    """
    translate string contents of list into contents without string
    e.g. function_list = ['function1', 'function2']
            -> translated_function_list = [function1, function2]

         input = 'hoge'
            -> translated_input = hoge

    :param  input  : input including string
    :return output : output excluding string
    """

    output_list = []

    if (type(input) is str):
        cmd = 'output_list.append(' + input + ')'
        exec(cmd)
        return output_list[0]
    elif (type(input) is list):
        if (len(input) == 0):
            return output_list
        else:
            for i in range(len(input)):
                cmd = 'output_list.append(' + input[i] + ')'
                exec(cmd)
            return output_list
    else:
        print('Please input str or "list" including "str".')


def save_result(sess, model, out_dirpath):
    """
    save training result (tf.Session and model)

    :param  sess       : tf.Session()
    :param  model      : d_model.Model()
    :param  out_dirpath: str, directory path to save results
    :return
    """
    if (not os.path.isdir(out_dirpath)):
        os.makedirs(out_dirpath)

    # save tf.Session()
    saver = tf.train.Saver()
    saver.save(sess, out_dirpath + 'model_sess.ckpt')

    # # save model
    # f = open(out_dirpath + 'model.textfile', 'w')
    # pickle.dump(model, f)
    # f.close
    pickle.dump(model, open(out_dirpath + 'model.pickle', 'wb'))
