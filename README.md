# cell-sequencing-neural-network
A cell sequencing neural network converted from Tensorflow to PyTorch.

1. Install the entire SummerProject file from the Dropbox.

2. In the terminal, run "bash /Users/[username]/Downloads/SummerProject/code/[name of Miniconda file]

    If you use an M1 Mac, download to arm_64 version. If your Mac has an Intel chip, download the x86 version.

3. Create a conda environment and activate it:
    conda create -n Cellcano python=3.8
    conda activate Cellcano

4. Download the necessary packages:
    pip install tensorflow
    pip install anndata
    pip install scanpy
    pip install keras
    pip install protobuf

    If it says that the requirement is already satisfied, ignore it and move on.

5. Run the main.py code:
    You may need to change some paths in the code, make sure to check it to make sure that any written paths will be able to find their corresponding files
    
    python /Users/[username]/Downloads/SummerProject/code/main.py
