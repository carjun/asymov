"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_mean_embeddings(bert, input_ids, attention_mask):
        bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output


def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(       #TODO: this tokenizes list or tuple
        text,                                       #TypeError: batch_text_or_text_pairs has to be a list or a tuple (got <class 'numpy.ndarray'>)
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return token_feat


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    for i, batch in enumerate(tqdm(train_loader)):
        # if i == 46:
        #      print(i)
        # text, label,  = batch['text'], batch['label']
        pose, ann  = batch['pose'].detach(), batch['annotation_emb'].detach()
        # num_ann_mask = num_ann>0
        # pose = pose[num_ann_mask]
        # ann = list(np.array(ann[0])[num_ann_mask])
        # if any(num_ann_mask):
        #     tokenized_features = get_batch_token(tokenizer, ann, max_length)
        #     corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)
            
        if i == 0:
            # all_labels = label
            # all_embeddings = corpus_embeddings.detach().numpy()
            all_embeddings = np.concatenate((pose.reshape(pose.shape[0], -1), ann), axis=1)

        else:
            # all_labels = torch.cat((all_labels, label), dim=0)
            batch_emb = np.concatenate((pose.reshape(pose.shape[0], -1), ann), axis=1)
            all_embeddings = np.concatenate((all_embeddings, batch_emb), axis=0)

    # Perform KMeans clustering
    # confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes, verbose=True)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    # true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    # print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))
    print("all_embeddings:{}, pred_labels:{}".format(all_embeddings.shape, len(pred_labels)))

    # confusion.add(pred_labels, true_labels)
    # confusion.optimal_assignment(num_classes)
    # print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))
    
    return clustering_model.cluster_centers_



