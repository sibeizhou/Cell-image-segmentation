cmpt742 Assignment2
============

This project implements the U-net architecture for cell image data segmentation using PyTorch

Requirements:
-------------
- Python 3.x
- All dependencies listed in `requirements.txt`

Running the Code:
-----------------
1. Set up the environment using `requirements.txt`.
2. Execute `run.sh` to run the code.

File Structure:
---------------
- data_processing.py: Preprocesses input data.
- model.py: Defines the machine learning model.
- train.py: Main training script.

materials
├── data
│   └── cells
│       ├── labels                # Folder for label images
│       └── scans                 # Folder for scan images
└── src
    ├── data_processing.py        # Contains data loader
    ├── model.py                  # Defines the Unet model
    ├── train.py                  # Main training script
    ├── README.txt                # Instructions and project description
    ├── requirements.txt          # The list of libraries
    └── run.sh                    # Command to run the code


