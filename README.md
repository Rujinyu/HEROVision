# HEROVision Model (last updated: 2024-06-21)

This repository contains python codes related HEROVision model, which shows promise in optimizing individualized treatment between TA and SR for early-stage rHCC, complementing current clinical guidelines.The training of the model was completed on NVIDIA GeForce GTX 3090.
- The data required to build the model **dataset.py**
-	Model architecture **Model_hovertrans.py**
- Training and evaluating surgical models **train_hover_SR.py**
- Training and evaluating ablation models **train_hover_TA.py**
- Some configuration information **config.py**

**To run the codes ensure you have the necessary prerequisites:**
-	Python installed on your system (version 3.7 or above).
-	Required libraries: torch, torchvision, numpy, SimpleITK, pydicom, lifelines, radiomics, matplotlib, and scipy. You can install them using pip:
`pip install torch torchvision numpy SimpleITK pydicom lifelines radiomics matplotlib and scipy`

**dataset.py**
1. Description: The dataset comprises four modalities of medical images:
    - Grayscale Ultrasound (US)
    - Contrast-Enhanced Ultrasound (CEUS)
    - T2-Weighted Imaging (T2WI)
    - Diffusion-Weighted Imaging (DWI)
2.	Number of Images: 15774 images.
3.	Label: PFS.
4.  Splitting: The datasets were divided into a training cohort and an internal validation cohort in a 4:1 ratio, based on chronological order. 
5. Resizing: All images are resized to a uniform dimension (e.g., 128*128 pixels) to standardize input for the neural network.
6. Normalization: Image pixel values are normalized to the range [0, 1] to facilitate faster convergence during training.

**Model_hovertrans.py**

**Architecture**: This architecture comprises four main components: 
  - The embedding layer, responsible for partitioning the original image into patches using diverse techniques; 
  - The feature extraction layer, leveraging the ViT model to extract and integrate features within each patch; 
  - Convolutional layers, which further integrate features extracted from patches using various partitioning methods; 
  - The Cox regression layer, utilizing the COX loss function to regress the extracted features and ultimately derive the corresponding risk value.
![Architecture](https://github.com/Rujinyu/HEROVision/blob/main/MODEL.jpg "Architecture")

**Hyperparameters**: During the training phase, we resized the images to 128 * 128 dimensions. We employed two stages with depths of 4 and 2, heads of 2 and 4, and dimensions of 4 and 8, respectively. The Cox loss function was selected. We utilized the Adam optimizer with a learning rate of 0.001 and a decay rate of 0.0001. Training was conducted over 100 epochs with a batch size of 16.

**train_hover_SR.py OR train_hover_TA.py**
- Description: 
- Training Procedure
  - **Optimizer**: The Adam optimizer is used for its efficiency in training deep learning models.
  - **Loss Function**: Cox Loss is used to regression time information.
  - **Callbacks**: Early stopping and model checkpointing are employed to prevent overfitting and save the best model during training.
- You can train the model use instruction:
`python train_hover_SR.py` or `python train_hover_TA.py`

## Future Work
- We plan to apply the HEROVision model in a prospective study to optimize personalized curative treatment for early-stage recurrent hepatocellular carcinoma.
