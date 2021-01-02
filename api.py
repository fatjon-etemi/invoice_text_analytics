import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import pickle
from pdf2image import convert_from_path
import pytesseract
import json
import pyodbc
import pandas as pd

ALLOWED_EXTENSIONS = {'pdf'}
config = json.load(open('config.json'))

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config["DEBUG"] = True
app.config["UPLOAD_FOLDER"] = './uploads'

model = pickle.load(open('model.pkl', 'rb'))


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
    if len(df) > 0:
        return df.loc[0].to_json()
    else:
        query = "SELECT {} FROM abasbi.dbo.Vendor$Vendorcontact_1_2 WHERE ID = '{}' AND MANDANT_ID = 2"
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
            predicted_id = model.predict([file_string]).item(0)
            supplier = json.loads(get_supplier(predicted_id))
            return render_template('result.html', supplier=supplier)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(port=3000)