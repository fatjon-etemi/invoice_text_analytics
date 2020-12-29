import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pickle
from pdf2image import convert_from_path
import pytesseract
from gensim.utils import simple_preprocess
import json
import pyodbc
import pandas as pd
from main import preprocess

ALLOWED_EXTENSIONS = {'pdf'}
config = json.load(open('config.json'))

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config["DEBUG"] = True
app.config["UPLOAD_FOLDER"] = './uploads'

model = pickle.load(open('doc2vec.pkl', 'rb'))
svm_model = pickle.load(open('svm.pkl', 'rb'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_supplier(id, fields = ['name']):
    server = config['server']
    database = config['database']
    username = config['db_user']
    password = config['db_password']
    cnxn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    query = "SELECT {} FROM abasbi.dbo.Vendor$Vendor_1_1 WHERE ID = '{}' AND MANDANT_ID = 2"
    query = query.format(','.join(fields), id)
    df = pd.read_sql(query, cnxn)
    return df.loc[0].to_json()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            images = convert_from_path(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_string = ''
            for image in images:
                file_string += pytesseract.image_to_string(image)
            vector = model.infer_vector(preprocess(file_string))
            predicted_id = svm_model.predict([vector]).item(0)
            return get_supplier(predicted_id)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(port=3000)