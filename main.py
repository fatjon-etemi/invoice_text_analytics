import os
import re
import shutil
import pyodbc
import pandas as pd
from pdf2image import convert_from_path
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pytesseract
import json
import matplotlib.pyplot as plt
import random

path = '//srvaw03/AF-Drive/BACH/Vendor/B'
pattern = '^PI20_[0-9]{6}$'
pattern1 = '^VE20_[0-9]*$'
data_fields = 'nummer,sumbrutto,sumnetto,lief'
data_path = './data'
config = json.load(open('config.json'))


def get_files():
    for folder in os.listdir(path):
        print(folder)
        if re.match(pattern, folder) and len(os.listdir(path + "/" + folder)) > 0 and folder not in os.listdir(data_path):
            for file in os.listdir(path + "/" + folder):
                os.mkdir(data_path + '/' + folder)
                shutil.copyfile(path + "/" + folder + "/" + file, data_path + "/" + folder + "/" + file)
        elif os.path.isdir(path + "/" + folder) and (re.match(pattern1, folder) or folder == 'invoices' or folder == '_archive'):
            get_files(path + "/" + folder)


def get_data():
    server = config['server']
    database = config['database']
    username = config['db_user']
    password = config['db_password']
    cnxn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    query = 'SELECT {} FROM abasbi.dbo.Purchasing$Purchasing_4_1 where MANDANT_ID = 2 AND NUMMER in ({})'
    numbers = ["'" + i[5:] + "'" for i in os.listdir(data_path)]
    query = query.format(data_fields, ','.join(numbers))
    df = pd.read_sql(query, cnxn)
    duplicates = set([x for x in df["nummer"].tolist() if df["nummer"].tolist().count(x) > 1])
    if len(duplicates) > 0:
        for folder in os.listdir(data_path):
            if folder[5:] in duplicates:
                shutil.rmtree(data_path + "/" + folder, ignore_errors=True)
        get_data()
    else:
        df.to_csv('data.txt')


def ocr():
    i = 0
    for folder in os.listdir(data_path):
        print(folder, len(os.listdir(data_path)) - i, 'remaining')
        i += 1
        if not os.path.isfile(data_path + '/' + folder + '/ocr.txt'):
            for file in os.listdir(data_path + "/" + folder):
                images = convert_from_path(data_path + "/" + folder + "/" + file)
                file_string = ''
                for image in images:
                    file_string += pytesseract.image_to_string(image)
                with open(data_path + "/" + folder + "/ocr.txt", "w") as f:
                    f.write(file_string)


def merge():
    df = pd.read_csv('data.txt')
    df = df.sort_values('nummer')
    text = []
    for folder in os.listdir(data_path):
        with open(data_path + "/" + folder + "/ocr.txt") as file:
            text.append(file.read())
    df['text'] = text
    df.to_csv('train.txt')


def clustering():
    train = pd.read_csv('train.txt')
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train["text"])]
    model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
    model.train(documents, total_examples=len(documents), epochs=10)
    number_of_clusters = len(set(train["lief"].tolist()))
    kmeans_model = KMeans(n_clusters=number_of_clusters, max_iter=100)
    X = kmeans_model.fit(model.docvecs.vectors_docs)
    labels = kmeans_model.labels_.tolist()
    print(labels)
    l = kmeans_model.fit_predict(model.docvecs.vectors_docs)
    pca = PCA(n_components=2).fit(model.docvecs.vectors_docs)
    datapoint = pca.transform(model.docvecs.vectors_docs)

    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(labels))]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
    plt.show()


if __name__ == '__main__':
    # get_files()
    # get_data()
    # ocr()
    # merge()
    clustering()