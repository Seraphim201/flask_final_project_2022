import numpy as np
from flask import Flask, render_template, request
import pickle
import sklearn
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)


def prediction(param):

    with open('models/model_d_dtr_best.pkl', 'rb') as f_d:
        loaded_model_d = pickle.load(f_d)
    with open('models/model_w_dtr_best.pkl', 'rb') as f_w:
        loaded_model_w = pickle.load(f_w)

    y_pred_d = loaded_model_d.predict(param)
    y_pred_w = loaded_model_w.predict(param)

    result = [y_pred_d, y_pred_w]

    return result


@app.route('/', methods=['POST', 'GET'])
def processing():

    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        iw_value = float(request.form.get('iw'))
        if_value = float(request.form.get('if'))
        vw_value = float(request.form.get('vw'))
        fp_value = float(request.form.get('fp'))

        param = np.array([[iw_value, if_value, vw_value, fp_value]])

        result = prediction(param)
        depth = result[0][0]
        width = result[1][0]
        return render_template('index.html', depth=depth, width=width)


if __name__ == '__main__':
    app.run()
