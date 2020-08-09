"""
Date: 8-Aug-2020
Author: Nitin Ashutosh <nitinashu1995@gmail.com>
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from extract_cnn_resnet50_keras import ResNet
class OffilinePhase:
    '''
    OFFLINE PHASE:
    Collecting feature vector for dataset
    '''
    def __inint__(self):
        self.img_list = None
    def get_imlist(self, data):
        '''
        Return list of images path from given dataset path.
        '''
        self.img_list = [os.path.join(data, f) for f in os.listdir(data) if f.endswith('.jpg')]
    def get_feature_vector(self):
        '''
        Save feature vector in .npy file
        '''
        print("--------------------------------------------------")
        print("         feature extraction starts")
        print("--------------------------------------------------")
        feats = []
        names = []
        resnet_obj = ResNet()
        resnet_obj.load_network()
        for i, img_path in enumerate(self.img_list):
            norm_feat = resnet_obj.extract_feat(img_path)
            img_name = os.path.split(img_path)[1]
            feats.append(norm_feat)
            names.append(img_name)
            print("extracting feature from image No. %d" %((i+1)))
        np.save('feats', feats)
        np.save('names', names)

class OnlinePhase():
    '''
    ONLINE PHASE:
    Getting similar images
    '''
    def _init__(self):
        self.img_query = None
        self.feats = None
        self.imgnames = None
    def load_feature(self):
        '''
        Loading saved feature vector which saved by OFFLINEPHASE
        '''
        self.feats = np.load('feats.npy')
        self.imgnames = np.load('names.npy')
        print("Image names: ", self.imgnames)
    def get_similar_img(self, img_query):
        '''
        Featute vector comparision.
        '''
        self.img_query = img_query
        resnet_obj = ResNet()
        resnet_obj.load_network()
        query_vec = resnet_obj.extract_feat(self.img_query)
        scores = np.dot(query_vec, self.feats.T)
        rank_id = np.argsort(scores)[::-1]
        rank_score = scores[rank_id]
        print(rank_id)
        print(rank_score)
        maxres = 9
        return [self.imgnames[index] for i, index in enumerate(rank_id[0:maxres])]


if __name__ == "__main__":
    ARGUMENT_PARSER = argparse.ArgumentParser()
    ARGUMENT_PARSER.add_argument("-dataset", required=True, help="Path to dataset")
    ARGUMENT_PARSER.add_argument("-img_query", required=True, help="image to be search")
    ARGS = vars(ARGUMENT_PARSER.parse_args())
    DATA = ARGS["dataset"]
    QUERY_IMG = ARGS["img_query"]
    #OFFLINE PHASE
    OFFLINEPHASE_OBJ = OffilinePhase()
    OFFLINEPHASE_OBJ.get_imlist(DATA)
    #Removig query image from image_list which will be used in feature extraction
    BASE_FILE_NAME = os.path.basename(QUERY_IMG)
    #OFFLINEPHASE_OBJ.img_list.remove(QUERY_IMG)
    if not (os.path.isfile("feats.npy") and os.path.isfile("names.npy")):
        OFFLINEPHASE_OBJ.get_feature_vector()
    else:
        print("##############################################")
        print("              Old features exits!!            ")
        print("##############################################")
        print("DO you want use old features? [Y/N]")
        RESPONSE = input()
        if RESPONSE == "N":
            OFFLINEPHASE_OBJ.get_feature_vector()
    #ONLINEPHSE
    ONLINEPHASE_OBJ = OnlinePhase()
    ONLINEPHASE_OBJ.load_feature()
    #Removing featue vector for query image
    INDEX = np.where(ONLINEPHASE_OBJ.imgnames == BASE_FILE_NAME)
    ONLINEPHASE_OBJ.imgnames = np.delete(ONLINEPHASE_OBJ.imgnames, INDEX)
    ONLINEPHASE_OBJ.feats = np.delete(ONLINEPHASE_OBJ.feats, INDEX, 0)
    RESULT_LIST = ONLINEPHASE_OBJ.get_similar_img(QUERY_IMG)
    FIG, AXARR = plt.subplots(4, 3)
    #plotting query image on first row and first coloumn
    AXARR[0, 0].imshow(mpimg.imread(QUERY_IMG))
    AXARR[0, 1].axis("off")
    AXARR[0, 2].axis("off")
    #plotting similar image from row 2-4
    LIST_IMAGE = []
    for i, img in enumerate(RESULT_LIST):
        LIST_IMAGE.append(mpimg.imread(DATA+"/"+img))
    COUNT = 0
    for row in range(1, 4, 1):
        for col in range(3):
            IMG = LIST_IMAGE[COUNT]
            AXARR[row, col].imshow(IMG)
            COUNT += 1
    plt.show()
    plt.close()
