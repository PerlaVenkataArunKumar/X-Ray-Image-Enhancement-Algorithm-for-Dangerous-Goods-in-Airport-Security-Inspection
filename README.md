# X-Ray Image Enhancement Algorithm for Dangerous Goods in Airport Security Inspection
## Project Overview
This project focuses on developing an X-Ray Image Enhancement Algorithm aimed at improving the detection of dangerous goods in airport security inspections. Leveraging advanced image enhancement techniques and machine learning models, the system enhances X-ray images for better visibility and identification of prohibited items.

The project utilizes Contrast Limited Adaptive Histogram Equalization (CLAHE) for image preprocessing, coupled with a Convolutional Neural Network (CNN) optimized using Particle Swarm Optimization (PSO). This ensures accurate and efficient detection of dangerous goods.

## Objectives
Enhance X-ray images to improve visibility and contrast of objects.
Accurately classify items in X-ray images into "safe" and "dangerous" categories.
Utilize Particle Swarm Optimization (PSO) for hyperparameter tuning to achieve optimal performance.
## Methodology
1. Image Preprocessing
Contrast Limited Adaptive Histogram Equalization (CLAHE): Enhances image contrast and highlights subtle differences in texture.
Normalization: Scales pixel values to a range of [0, 1] for consistency in model training.
Noise Reduction: Removes artifacts and noise to improve image clarity.
2. Convolutional Neural Network (CNN)
## CNN Architecture:
Input Layer: Processes 64x64 grayscale images.
## Convolutional Layers:
Layer 1: 32 filters, kernel size (3, 3), activation: ReLU.
Layer 2: 64 filters, kernel size optimized by PSO, activation: ReLU.
### Pooling Layers: MaxPooling with a pool size of (2, 2) to reduce dimensions.
## Fully Connected Layers:
Dense Layer: 128 neurons, activation: ReLU.
Output Layer: 2 neurons (safe/dangerous), activation: Softmax.
Optimizer: Adam.
## Loss Function: Binary Crossentropy.
3. Particle Swarm Optimization (PSO)
## PSO Process:
Representation: Particles represent hyperparameters: [filters_1, filters_2, learning_rate, batch_size].
Fitness Evaluation: Validation accuracy of the CNN model trained with particle parameters.
## Swarm Update:
Particles adjust positions using velocity based on personal best and global best positions.
Iteration: Repeated over several generations to identify optimal parameters.
4. Integration and Training
Model Training: CNN is trained using hyperparameters provided by PSO.
Validation Accuracy: Determines the fitness score of each particle.
Final Model: Best hyperparameters are used to train the model for 20 epochs.
## Results
### Optimal Hyperparameters:
Filters (Layer 1): 32
Filters (Layer 2): 64
### Learning Rate: 0.0005
Batch Size: 64
### Performance Metrics:
Validation Accuracy: 95%

Test Accuracy: 93%
