#!/bin/bash
 
if ls *.db 1> /dev/null 2>&1; then
    echo "database exists."
else
    echo "creating database"
    python utils/clean_csv.py
    python utils/db_setup.py
fi
