# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 21:30:22 2021

@author: Erick Muñiz Morales

Problem 2. From a clinical trial, we have 12 patients with HIV infection. After treatment, the disease
progressed in 6 patients (1) and in 6 patients the infection did not progress (0). Four measurements
are taken in the 12 patients (Age, sugar levels, T cell levels and Cholesterol). Which measurement
can be used as a marker to describe progression of the disease? Which will be the criteria to predict
the progression? The data can be found in „problem2.csv (x_age, x_sugar, x_Tcell, x_cholesterol,
outcome). Arrange the data and briefly explain your results. The variable “y” (target) is a vector of 0
and 1 to represent the progression.
"""
import pandas as pd 
