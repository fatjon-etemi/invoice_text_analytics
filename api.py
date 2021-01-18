import os
from flask import Flask, flash, request, redirect, render_template, Markup, send_from_directory
from werkzeug.utils import secure_filename
import pickle
from pdf2image import convert_from_path
import pytesseract
import json
import pyodbc
import pandas as pd
import re
from datetime import datetime
import glob
import random

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


def get_supplier(id, fields=['name']):
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


def extract_data(label, text):
    specific_regex_file = "./regex_templates/" + label + ".json"
    template_regex_file = "./regex_templates/(TEMPLATE).json"
    if os.path.isfile(specific_regex_file):
        regex_template = json.load(open(specific_regex_file))
    else:
        regex_template = json.load(open(template_regex_file))
    result = {}
    for k, v in regex_template.items():
        if k == 'options':
            continue
        x = re.findall(v, text, re.MULTILINE)
        if len(x) > 0:
            if type(x[-1]) == str:
                result[k] = x[-1]
            else:
                result[k] = list(filter(None, x[-1]))[0]
        # if len(x) > 0:
        #     list(filter('', x))[0]
    if 'options' in regex_template:
        options = regex_template['options']
        if 'invoice_date' in result and 'date_format' in options:
            result['invoice_date_unformated'] = result['invoice_date']
            result['invoice_date'] = datetime.strptime(result['invoice_date'], options['date_format']).strftime(
                config['standard_dateformat'])
        if 'split' in options:
            for k in options["split"]:
                v = options["split"][k]
                if k in result:
                    splt = result[k].split(v[0])
                    if len(splt) > 1:
                        result[k] = splt[v[1]]
    for k, v in result.items():
        text = text.replace(v, '<span class="highlight">%s</span>' % v, 1)
    return regex_template, text, result


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
            regex_template, text, data = extract_data(predicted_id, file_string)
            data['supplier_id'] = predicted_id
            data['supplier_name'] = supplier['name']
            if request.args.get('format') == 'json':
                return json.dumps(data)
            return render_template('result.html', data=data, text=Markup(text), form_data=regex_template,
                                   pdf_file=os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template("index.html")


@app.route('/regex', methods=['POST'])
def update_regex_file():
    if request.method == 'POST':
        my_dict = {'invoice_number': request.form['invoice_number'], 'amount': request.form['amount'],
                   'invoice_date': request.form['invoice_date'], 'currency': request.form['currency']}
        if 'options' in request.form:
            my_dict['options'] = json.loads(request.form['options'])
        print(my_dict)
        with open('./regex_templates/' + request.form['regex_template_name'], 'w') as file:
            file.write(json.dumps(my_dict, indent=2))
        return 'Saved!'


@app.route('/random', methods=['GET'])
def random_invoice():
    files = glob.glob('./data/*/*.pdf')
    file = random.choice(files)
    images = convert_from_path(file)
    file_string = ''
    for image in images:
        file_string += pytesseract.image_to_string(image)
    predicted_id = model.predict([file_string]).item(0)
    supplier = json.loads(get_supplier(predicted_id))
    regex_template, text, data = extract_data(predicted_id, file_string)
    data['supplier_id'] = predicted_id
    data['supplier_name'] = supplier['name']
    if 'options' in regex_template:
        regex_template['options'] = json.dumps(regex_template['options'])
    return render_template('result.html', data=data, text=Markup(text), form_data=regex_template, pdf_file=file)


@app.route('/data/<path:path>', methods=['GET'])
def send_file_data(path):
    return send_from_directory('data', path)


@app.route('/uploads/<path:path>', methods=['GET'])
def send_file_uploads(path):
    return send_from_directory('uploads', path)


@app.route('/book', methods=['POST'])
def book():
    return 'booked!'


if __name__ == '__main__':
    app.run(port=3000)
