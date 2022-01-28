#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import os
from os.path import basename as ospb
from os.path import join as ospj

import numpy as np
import pandas as pd
import pdb


"""
Script to save the sequence of symbols for each sequence in the dataset.
"""

D_FOLDER = '/ps/project/conditional_action_gen/asymov/kit-molan/dataset/'

def read_rifke(f):
    '''Return RIFKE contents of file'''
    df_rifke = pd.read_csv(f)
    return df_rifke


def get_symbols(rifke):
    '''Given the rifke file contents, return seq. of symbols.'''
    T = len(rifke)

    # Stub. TODO: Implement actual code.
    l_sym = 33 + np.zeros(T, dtype=int)
    return l_sym


def write_symbols(l_sym, sym_fn):
    '''Save seq. of symbols as a csv file of 'appropriate' format.'''
    # Each row of df = symbol for i-th index/frame
    df = pd.DataFrame(l_sym, columns=['symbol'])
    df.to_csv(sym_fn)
    return None


def add_symbol_data_to_dataset():

    # Get paths for all files
    df = pd.read_hdf('./data.h5', 'df')
    l_fns = df['rifke']
    l_sym_fns = []

    # For each seq. in dataset
    for i, f in enumerate(l_fns):

        # Find corresponding symbol file
        rifke_fn = ospj(D_FOLDER, ospb(f))
        sym_fn = rifke_fn.replace('rifke', 'sym')

        if not os.path.exists(sym_fn):
            # get symbols
            rifke = read_rifke(rifke_fn)
            l_sym = get_symbols(rifke)

            # store in file
            write_symbols(l_sym, sym_fn)
            l_sym_fns.append(sym_fn)

    # Save new dataframe. Commented for now to prevent accidental overwrite.
    df['symbol'] = l_sym_fns
    # df.to_hdf('./data.h5', key='df', mode='w')

add_symbol_data_to_dataset()
