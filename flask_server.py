import flask
from flask import jsonify
import json
from flask import Flask, request, Response

app = Flask(__name__)
HOST = "0.0.0.0"
W = 640
H = 480
# W = 1920
# H = 1080


# W = 1920
# H = 1080
# Testing URL
@app.route('/api/v1/config/', methods=['GET'])
def hello_world():
    ret = {"code": 200, "data": {"X": W, "Y": H}, "success": True}
    return Response(json.dumps(ret, ensure_ascii=False),
                    mimetype='application/json')


app.run(debug=True, port=8080, host=HOST)
