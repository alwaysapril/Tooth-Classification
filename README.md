# Teeth Project

## Setup Guide

### Requirements

- Python version: **3.11.10**

### Installation Steps

1. Clone the GitHub repository:
   ```sh
   git clone https://FardinMultani@bitbucket.org/piyushkshatra/teeth.git
   ```
2. Check out the branch `dev_main`.
3. Change to the project directory:
   ```sh
   cd teeth
   ```
4. Create a virtual environment:
   ```sh
   python -m venv teeth_env
   ```
5. Activate the virtual environment:
   - On Windows:
     ```sh
     teeth_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source teeth_env/bin/activate
     ```
6. Install the required dependencies from `requirements.txt` (located in the `dev_main` branch):
   ```sh
   pip install -r requirements.txt
   ```
7. Download Model weight from this link:
   ```sh
   https://drive.google.com/file/d/1OpNZ3_dsqPXzvV0hm1J-8RhkJEQn0gYo/view?usp=sharing
   ```
Download model `.pth` file from here.

8. create `models' folder(directory) like below structure and Put model in that directory ,
    ```sh
   teeth/models/put .pth model here.
   ```


### Running the Inference Pipeline

To run the inference pipeline:

1. Execute the `inference.py` script in the terminal:
   ```sh
   python inference.py
   ```
2. When prompted, paste the path to the `.stl` file in the terminal and press **Enter**.
3. The script will display the **Top-3 classes** along with their **probabilities**.

---

## Model Training Details

- **Training Data Accuracy**: 90.76%
- **Training Data Loss**: 0.0593

- **Hyper Parameter for Train model**:
    - Deep Learning Model :- Effficent_net Architecture(3D)
    - num_epochs: 100
    - batch_size:10
    - Optimizer : Adam
    - no of classes : 8
    - Loss Function : Cross Entropy loss


1. You can see Loss grpah accroding to batch and epochs wise here this below folder path:
   ```sh
   teeth/loss_graphs
   ```


This project provides a pipeline for teeth classification based on `.stl` files. Ensure all dependencies are installed before running the inference script.
