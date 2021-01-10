import json
import os
import re
import shutil
from tqdm import tqdm

import pandas as pd
import pyodbc
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError

path = '//srvaw03/AF-Drive/BACH/Vendor/B'
pattern = '^PI20_[0-9]{6}$'
pattern1 = '^VE20_[0-9]*$'
data_fields = 'nummer,sumbrutto,sumnetto,lief2'
data_path = './data'
config = json.load(open('config.json'))


def get_files(path):
    for folder in os.listdir(path):
        print(folder)
        if re.match(pattern, folder) and len(os.listdir(path + "/" + folder)) > 0 and folder not in os.listdir(
                data_path):
            for file in os.listdir(path + "/" + folder):
                os.mkdir(data_path + '/' + folder)
                shutil.copyfile(path + "/" + folder + "/" + file, data_path + "/" + folder + "/" + file)
        elif os.path.isdir(path + "/" + folder) and (
                re.match(pattern1, folder) or folder == 'invoices' or folder == '_archive'):
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
    t = tqdm(os.listdir(data_path))
    for folder in t:
        t.set_description(folder)
        t.refresh()
        # print(folder, len(os.listdir(data_path)) - i, 'remaining')
        if not os.path.isfile(data_path + '/' + folder + '/ocr.txt'):
            for file in os.listdir(data_path + "/" + folder):
                try:
                    images = convert_from_path(data_path + "/" + folder + "/" + file)
                    file_string = ''
                    for image in images:
                        file_string += pytesseract.image_to_string(image)
                    with open(data_path + "/" + folder + "/ocr.txt", "w") as f:
                        f.write(file_string)
                except PDFPageCountError:
                    continue


def merge():
    df = pd.read_csv('data.txt')
    df = df.sort_values('nummer')
    text = []
    for folder in os.listdir(data_path):
        with open(data_path + "/" + folder + "/ocr.txt") as file:
            text.append(file.read())
    df['text'] = text
    df.to_csv('train.txt')


if __name__ == '__main__':
    # get_files(path)
    ocr()
    get_data()
    merge()
