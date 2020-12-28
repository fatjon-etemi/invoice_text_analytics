import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(port=3000)