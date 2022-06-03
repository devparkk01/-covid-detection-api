import flask
from flask import Flask
import  json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def getter():
    return json.dumps({"key": 11})

if __name__=="__main__":
    app.run(host='0.0.0.0', port = 5000)