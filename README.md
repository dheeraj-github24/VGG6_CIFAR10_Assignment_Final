This project trains a VGG6 convolutional neural network on the CIFAR-10 dataset using PyTorch and Weights & Biases (W&B) for experiment tracking and hyperparameter optimization.
The implementation supports data augmentation, model evaluation, and sweep-based hyperparameter tuning, and saves the best models automatically.

Google Colab:
The entire workflow was developed and tested in Google Colab using GPU acceleration (T4/A100).

Library Versions:

- Python	3.10
- PyTorch	2.x
- Torchvision	0.19.x
- W&B	0.17.x
- NumPy	1.26.x
- CUDA	12.1 (Colab default)

Project Structure:

- Cell 1 – Setup & Install	    Installs required dependencies and authenticates W&B.
- Cell 2 – Dataset Preparation	Loads CIFAR-10 with normalization and augmentation.
- Cell 3 – Model Definition	    Defines modular VGG6 model with configurable activation.
- Cell 4 – Training & Evaluation	Handles model training, validation, and W&B logging.
- Cell 5 – Sweep Execution	    Runs random hyperparameter search across activations, optimizers, etc.
- Cell 6 – Model Export	        Saves best models (.pth) and creates a downloadable ZIP file.

How To Run?

- This project was implemented and executed using Google Colab.
- Please refer to the assignment_1.py file for the complete workflow.
- There are 6 execution steps — each corresponding to a logical section of the code.
- To reproduce the results:
        - Open the notebook in Google Colab.
        - Copy the code from each step in assignment_1.py into separate cells.
        - Run each cell sequentially (from Step 1 to Step 6) to ensure proper data loading, model training, and evaluation. While running it asks for the API key                        
           of WandB, please provide your key for execution.
        - Each step is clearly named and structured for clarity and reproducibility.
