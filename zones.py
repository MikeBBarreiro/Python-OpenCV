import sys
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

class user_app_callback_class():
    def __init__(self):    
        print(sys.executable)

if __name__ == "__main__":
    user_prompt = input("Type 'run' for results: ")
    
    if(user_prompt == 'run'):
        user_data = user_app_callback_class()