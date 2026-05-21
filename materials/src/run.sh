#!/bin/bash

if [ ! -d "env" ]; then
    python -m venv new_env
fi

source env/bin/activate

pip install -r requirements.txt

python train.py

deactivate
