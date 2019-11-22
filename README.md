# Toxicity-Detection

## Setting up the data from Kaggle on local machine

1. Download Kaggle.json by going to your Kaggle account settings and clicking on *Create New API Token*.
    Put Kaggle.json into your project directory

2. Run the following code in unix/bash terminal (Assuming in project directory)
    ```console
    pip install -q kaggle 
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
    unzip train.csv.zip -d ./data
    unzip test_private_expanded.csv.zip -d ./data
    unzip test_public_expanded.csv.zip -d ./data
    ```
    Now the data should be under the folder directory called *data*.
    
3. Run main.py or convert it into a jupyter notebook to see the data
   
