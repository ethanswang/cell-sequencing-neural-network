import os
import anndata

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from models.MLP_torch import MLP
import torch
import torch.nn as nn
import torch.optim as optim

## import my written library
from utils import _utils

if __name__ == '__main__':
    ## load train_adata and test_adata
    adata = anndata.read_h5ad("/Users/zhiyong/Downloads/SummerProject/data/Mouse_wholebrain_FC.h5ad")
    train_adata = adata[adata.obs['ind'] == 'P60FCRep1']
    test_adata = adata[adata.obs['ind'] == 'P60FCRep2']

    ## extract common genes first
    common_genes = set(train_adata.var_names).intersection(set(test_adata.var_names))
    train_adata = train_adata[:, list(common_genes)]
    test_adata = test_adata[:, list(common_genes)]

    ## preprocess the training data
    train_adata = _utils._process_adata(train_adata, process_type='train')
    train_adata = _utils._select_feature(train_adata, 
                fs_method='F-test', num_features=1000) ## use F-test to select 1000 informative genes
    train_adata = _utils._scale_data(train_adata) ## center-scale
    #_utils._visualize_data(train_adata, output_dir=".", prefix="traindata_vis")  ## visualize cell types with selected features on a low dimension (you might need to change some parameters to let them show all the cell labels)
    ## train an MLP model on it
    MLP_DIMS = _utils.MLP_DIMS ## get MLP structure from _utils.py
    x_train = _utils._extract_adata(train_adata)
    enc = OneHotEncoder(handle_unknown='ignore')
    y_train = enc.fit_transform(train_adata.obs[[_utils.Celltype_COLUMN]]).toarray()
    
    mlp = MLP()
    mlp.train()
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = mlp(torch.from_numpy(x_train))
        loss = criterion(outputs, y_train)
        loss.backward()
        print(f'Epoch {epoch} loss : {loss:.3f}')
        optimizer.step()
    '''
    mlp = _utils._init_MLP(x_train, y_train, dims=MLP_DIMS,
            seed=_utils.RANDOM_SEED)
    print("Printing out the model structure...")
    print(mlp.model.summary())
    mlp.compile()
    mlp.fit(x_train, y_train)
    mlp.model.save('./trained_MLP')  ## save the model so that you can load and play with it
    encoders = dict()
    for idx, cat in enumerate(enc.categories_[0]):
        encoders[idx] = cat

    ## preprocess the test data and predict cell types
    test_adata = _utils._process_adata(test_adata, process_type='test')
    test_adata = test_adata[:, list(train_adata.var_names)]  ## extract out the features selected in the training dataset
    test_data_mat = _utils._extract_adata(test_adata)
    test_data_mat = (test_data_mat - np.array(train_adata.var['mean']))/np.array(train_adata.var['std'])
    y_pred = tf.nn.softmax(mlp.model.predict(test_data_mat)).numpy()
    pred_celltypes = _utils._prob_to_label(y_pred, encoders)
    test_adata.obs[_utils.PredCelltype_COLUMN] = pred_celltypes

    ## let us evaluate the performance --> luckily you will have the accuracy over 99%
    from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
    print("Overall Accuracy:", accuracy_score(test_adata.obs[_utils.Celltype_COLUMN], test_adata.obs[_utils.PredCelltype_COLUMN]))
    print("ARI:", adjusted_rand_score(test_adata.obs[_utils.Celltype_COLUMN], test_adata.obs[_utils.PredCelltype_COLUMN]))
    print("Macro F1:", f1_score(test_adata.obs[_utils.Celltype_COLUMN], test_adata.obs[_utils.PredCelltype_COLUMN], average='macro'))
    ## a lot more evaluation metrics can be found on sklearn.metrics and you can explore with them 
    '''



