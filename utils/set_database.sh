#!/bin/bash
 
python utils/clean_csv.py
python utils/db_population.py
python utils/db_setup.py
