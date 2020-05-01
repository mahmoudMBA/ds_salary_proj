#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:26:17 2020

@author: mahmoud
"""

import requests
from data_input import data_in
URL='http://0.0.0.0:8080/predict'
#URL='http://127.0.0.1:5000/predict'
headers={"Content-Type":"application/json"}
data={'input':data_in}

r=requests.get(URL,headers=headers,json=data)

r.json()