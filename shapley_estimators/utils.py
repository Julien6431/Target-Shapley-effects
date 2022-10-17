#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:37:37 2021

@author: Julien Demange-Chryst
"""

#%% Module

import itertools

#%% Functions

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
