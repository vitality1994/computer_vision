#%%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath



#%%
def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list





def compute_dsift(img, stride, size):

    # To do

    sift = cv2.SIFT_create()
    list_kps = []

    for one_y in range(0, img.shape[0], stride):
        for one_x in range(0, img.shape[1], stride):
            list_kps.append(cv2.KeyPoint(one_x, one_y, size))

    _, dense_feature = sift.compute(img, list_kps)

    return dense_feature





def get_tiny_image(img, output_size):
    
    # To do

    resized_img = cv2.resize(img, (output_size[0], output_size[1]))
    feature = (resized_img - np.mean(resized_img)) / np.std(resized_img)

    return feature





def predict_knn(feature_train, label_train, feature_test, k):

    # To do

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(feature_train)

    label_test_pred = []

    neigh_idx = neigh.kneighbors(feature_test)[1]

    label_test_pred = []
    
    for one_row_indicies in neigh_idx:

        single_row_list = []

        for idx in one_row_indicies:

            single_row_list.append(label_train[idx])


        unique_indxs = sorted(set(single_row_list))

        frequencies = {}

        for x in unique_indxs:
            frequencies[x]=single_row_list.count(x)

        max_label = max(frequencies, key=frequencies.get)


        label_test_pred.append(max_label)


    label_test_pred = np.array(label_test_pred)


    return label_test_pred





def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    
    # To do

    list_train_tiny_imgs = [get_tiny_image(cv2.imread(one_img, 0), (16, 16)).flatten() for one_img in img_train_list]
  
    list_train_tiny_imgs = np.array(list_train_tiny_imgs)
    list_train_labels = np.array(label_train_list)


    list_test_tiny_imgs = [get_tiny_image(cv2.imread(one_img, 0), (16, 16)).flatten() for one_img in img_test_list]
    
    list_test_tiny_imgs = np.array(list_test_tiny_imgs)
    list_test_labels = np.array(label_test_list)  


    knn_test_preds = predict_knn(list_train_tiny_imgs, list_train_labels, list_test_tiny_imgs, 10)

    
    confusion = []
    accuracies_list = []

    for one_class_1 in label_classes:
        
        one_row = []

        first_class_indicies = np.where(knn_test_preds==one_class_1)

        g_truths = list_test_labels[first_class_indicies]

        for one_class_2 in label_classes:

            one_row.append(len(np.where(g_truths==one_class_2)[0]) / (len(g_truths)+0.0000000000001))
            

            if one_class_1 == one_class_2:

                accuracies_list.append(len(np.where(g_truths==one_class_2)[0]) / (len(g_truths)+0.0000000000001))


        confusion.append(one_row)

    confusion = np.array(confusion)
    accuracy = np.mean(accuracies_list)

    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy





def build_visual_dictionary(dense_feature_list, dict_size):

    # To do
    

    temp = np.array(dense_feature_list)

    temp = temp.reshape(len(temp), -1).astype(float)

    kmeans = KMeans(n_clusters=dict_size, max_iter=1).fit(temp)

    vocab = kmeans.cluster_centers_


    return vocab





def compute_bow(feature, vocab):
    
    # To do

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vocab)

    neigh_idx = neigh.kneighbors(feature)[1]


    hist = np.histogram(neigh_idx, bins=len(vocab))[0]

    normalized_hist = [float(i)/sum(hist)+0.0000000000001 for i in hist]
    

    bow_feature = np.array(normalized_hist)

    return bow_feature





def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    # To do

    try:
        vocab = np.loadtxt('/Users/jooyong/github_locals/computer_vision/cv_hw3/vocab_80_5_5_for_knn.txt')
    
    except:
        print("Let's make dictionary.")
        list_descs  = []

        for single_img in img_train_list:
            
            single_img = cv2.imread(single_img, 0)

            descs = compute_dsift(single_img, 5, 5)
            
            list_descs+=list(descs)
        
        vocab = build_visual_dictionary(list_descs, 80)

        np.savetxt(f'vocab_{80}_{5}_{5}_for_knn.txt', vocab)


    # making train/test set for knn

    list_train_bow = []
    list_train_labels = []

    for one_img, one_label in zip(img_train_list, label_train_list):
        
        one_bow = compute_bow(compute_dsift(cv2.imread(one_img, 0), 5, 5), vocab)

        list_train_bow.append(one_bow)
        list_train_labels.append(one_label)
    

    list_train_bow = np.array(list_train_bow)
    list_train_labels = np.array(list_train_labels)


    list_test_bow = []
    list_test_labels = []


    for one_img, one_label in zip(img_test_list, label_test_list):
    

        one_bow = list(compute_bow(compute_dsift(cv2.imread(one_img, 0), 5, 5), vocab))


        if np.isnan(one_bow).any()!=True:

            list_test_bow.append(one_bow)
            list_test_labels.append(one_label)
    
    list_test_bow = np.array(list_test_bow)
    list_test_labels = np.array(list_test_labels)


    knn_test_preds = predict_knn(list_train_bow, list_train_labels, list_test_bow,  15)
  

    confusion = []
    accuracies_list = []


    for one_class_1 in label_classes:
        
        one_row = []

        first_class_indicies = np.where(knn_test_preds==one_class_1)


        g_truths = list_test_labels[first_class_indicies]

        for one_class_2 in label_classes:

            if len(g_truths)==0:

                one_row.append(0)

            else:

                one_row.append(len(np.where(g_truths==one_class_2)[0]) / (len(g_truths)))
            

            if one_class_1 == one_class_2:

                if len(g_truths)==0:

                    accuracies_list.append(0)

                else:
                    
                    accuracies_list.append(len(np.where(g_truths==one_class_2)[0]) / (len(g_truths)))    


        confusion.append(one_row)



    confusion = np.array(confusion)
    accuracy = np.mean(accuracies_list)





    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy






def predict_svm(feature_train, label_train, feature_test, c):

   
    # To do

    clf =  LinearSVC(C=c)
    clf.fit(feature_train, label_train)

    label_test_pred = []

    
    for one_feature in feature_test:

        pred_label = clf.predict(np.array(one_feature).reshape(1, -1))

        label_test_pred.append(pred_label[0])


    label_test_pred = np.array(label_test_pred)


    return label_test_pred






def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    # To do
        
    try: 
        vocab = np.loadtxt('/Users/jooyong/github_locals/computer_vision/cv_hw3/vocab_100_5_5_for_svm.txt')

    except:
        print("Let's make dictionary.")
        list_descs  = []


        for single_img in img_train_list:
            
            single_img = cv2.imread(single_img, 0)

            descs = compute_dsift(single_img, 5, 5)
        

            list_descs+=list(descs)
        

        vocab = build_visual_dictionary(list_descs, 100)

        np.savetxt(f'vocab_{100}_{5}_{5}_for_svm.txt', vocab)


    # making train/test set for knn

    list_train_bow = []
    list_train_labels = []

    for one_img, one_label in zip(img_train_list, label_train_list):
        

        one_bow = compute_bow(compute_dsift(cv2.imread(one_img, 0), 5, 5), vocab)

            
        list_train_bow.append(one_bow)
        list_train_labels.append(one_label)
    
    list_train_bow = np.array(list_train_bow)
    list_train_labels = np.array(list_train_labels)


    list_test_bow = []
    list_test_labels = []


    for one_img, one_label in zip(img_test_list, label_test_list):
    
        one_bow = list(compute_bow(compute_dsift(cv2.imread(one_img, 0), 5, 5), vocab))

        if np.isnan(one_bow).any()!=True:

            list_test_bow.append(one_bow)
            list_test_labels.append(one_label)
    
    list_test_bow = np.array(list_test_bow)
    list_test_labels = np.array(list_test_labels)


    svm_test_preds = predict_svm(list_train_bow, list_train_labels, list_test_bow, 22)
  

    confusion = []
    accuracies_list = []


    for one_class_1 in label_classes:
        
        one_row = []

        first_class_indicies = np.where(svm_test_preds==one_class_1)


        g_truths = list_test_labels[first_class_indicies]

        for one_class_2 in label_classes:

            if len(g_truths)==0:

                one_row.append(0)

            else:

                one_row.append(len(np.where(g_truths==one_class_2)[0]) / (len(g_truths)))
            

            if one_class_1 == one_class_2:

                if len(g_truths)==0:

                    accuracies_list.append(0)

                else:
                    
                    accuracies_list.append(len(np.where(g_truths==one_class_2)[0]) / (len(g_truths)))    


        confusion.append(one_row)


    confusion = np.array(confusion)
    accuracy = np.mean(accuracies_list)


    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy




def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()




#%%
if __name__ == '__main__':
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

# %%
