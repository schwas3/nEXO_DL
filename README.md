# nEXO_DL
This is the package developed for DNN-based energy reconstruction and signal/background separation in the nEXO experiment. The package is built on pytorch platform. It takes the nEXO charge simulation as input (possibly adding photon information in the future), and utilizes a 18-layer ResNet for classification and AlexNet for channel charge reconstruction..

## Content of the package
 1. ~nEXO2DChargeImage.py - script to convert nEXO charge simulation to two images. Only two channels of the image are currently used. The third channel is open for future addition of photon information.~ This has been replaced with DnnEventTagger in nexo-offline.
 2. image2dcharge_csv.py - script to build csv file for dataset build.
 3. nEXODataset.py - main script for deep learning event classification model construction, training, and testing.
 4. resnet_example.py - ResNet configuration file. copied from https://github.com/DeepLearnPhysics/pytorch-resnet-example
 5. nEXO2DChargeImage_channelQ.py - script to convert nEXO charge simulation saved in ROOT to numpy arrays saved in npy file.
 6. image2dcharge_regression.py - script to build csv file for dataset used in channel charge reconstruction.
 7. nEXORegression.py - main script for deep learning channel charge reconstruction model construction, training and testing. 
## Requirements
 * pytorch(1.3.1), pandas(0.23.4), scipy(1.2.1), numpy(1.16.6)  
 * ROOT6
 * dataset - The nEXO simulation and simulation result is not open. 
 
