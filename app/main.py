from flask import Flask, request
from flask_cors import CORS
import pickle
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import base64
from io import BytesIO



### Load Models

## Malaria
#Load Model
interpreter_malaria = tflite.Interpreter(model_path='models/malaria.tflite')
interpreter_malaria.allocate_tensors()
#get input and output tensors
input_details = interpreter_malaria.get_input_details()
output_details = interpreter_malaria.get_output_details()

## Pneumonia
# Load Model
interpreter_pneumonia = tflite.Interpreter(model_path='models/pneumonia.tflite')
interpreter_pneumonia.allocate_tensors()
#get input and output tensors
input_details = interpreter_pneumonia.get_input_details()
output_details = interpreter_pneumonia.get_output_details()

## Diabetes
# Load Model
model_diabetes = pickle.load(open('models/diabetes.pkl','rb'))

## Breast Cancer
# Load Model
model_breast = pickle.load(open('models/breast_cancer.pkl','rb'))

## Heart
# Load Model
model_heart = pickle.load(open('models/heart.pkl','rb'))

## Kidney
# Load Model
model_kidney = pickle.load(open('models/kidney.pkl','rb'))

## Liver
# Load Model
model_liver = pickle.load(open('models/liver.pkl','rb'))


# Define App
app = Flask(__name__)
CORS(app)


def predict(values):
    if len(values) == 8:
        values = np.asarray(values)
        return model_diabetes.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        values = np.asarray(values)
        return model_breast.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        values = np.asarray(values)
        return model_heart.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        values = np.asarray(values)
        return model_kidney.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        values = np.asarray(values)
        return model_liver.predict(values.reshape(1, -1))[0]
    else:
        return -1

@app.route("/", methods = ['POST', 'GET'])
def API():
    pred = 0

    if request.method == 'POST':
        try:
            JSON = request.get_json()

            if JSON == None:
                return {'output': 'Error', 'msg': 'No data'}, 200
            elif JSON.get('image', None) == "malaria":
                if JSON.get('imageData', None) != None:
                    data_url = JSON['imageData'].split(",")[1]
                    img_bytes = base64.b64decode(data_url, ' /')
                    img = Image.open(BytesIO(img_bytes))
                    img = img.resize((36, 36))
                    img = np.asarray(img)

                    if img.shape[2] == 4:
                        img = img[:, :, :3]

                    img = img.reshape((1,36,36,3))
                    img = img.astype(np.float32)

                    #get prediction
                    interpreter_malaria.set_tensor(input_details[0]['index'], img)
                    interpreter_malaria.invoke()
                    output_data = interpreter_malaria.get_tensor(output_details[0]['index'])
                    pred = np.argmax(output_data[0])

                    return {'output': 'Sucess', 'msg': str(pred)}, 200
                else:
                    print('No Image Found')
                    return {'output': 'Error', 'msg': 'No Image Data Found'}, 200
            elif JSON.get('image', None) == "pneumonia":
                if JSON.get('imageData', None) != None:
                    data_url = JSON['imageData'].split(",")[1]
                    img_bytes = base64.b64decode(data_url, ' /')
                    img = Image.open(BytesIO(img_bytes))
                    img = img.resize((36, 36))
                    img = np.asarray(img)
                    img = img.reshape((1,36,36,1))
                    img = img.astype(np.float32)
                    img = img / 255.0
                    
                    #get prediction
                    interpreter_pneumonia.set_tensor(input_details[0]['index'], img)
                    interpreter_pneumonia.invoke()
                    output_data = interpreter_pneumonia.get_tensor(output_details[0]['index'])
                    pred = np.argmax(output_data[0])

                    return {'output': 'Sucess', 'msg': str(pred)}, 200
                else:
                    print('No Image Found')
                    return {'output': 'Error', 'msg': 'No Image Data Found'}, 200
            elif JSON.get('data', None) != None:
                to_predict_list = list(map(float, list(JSON.get('data').values())))
                pred = predict(to_predict_list)
                
                if pred == -1:
                    return {'output': 'Error', 'msg': 'Wrong Data or Insufficient Dara'}, 200
                else:
                    return {'output': 'Sucess', 'msg': str(pred)}, 200
            else:
                return {'output': 'Error', 'msg': 'No Data or Wrong Data'}
        except Exception as ex:
            print (ex)
            return {'output': 'Error', 'msg': 'Unknown Error'}
    else:
        try:
            JSON = request.args

            if JSON == None:
                return {'output': 'Error', 'msg': 'No data'}, 200
            elif JSON.get('image', None) == "malaria":
                if JSON.get('imageData', None) != None:
                    data_url = JSON['imageData'].split(",")[1]
                    img_bytes = base64.b64decode(data_url, ' /')
                    img = Image.open(BytesIO(img_bytes))
                    img = img.resize((36, 36))
                    img = np.asarray(img)

                    if img.shape[2] == 4:
                        img = img[:, :, :3]

                    img = img.reshape((1,36,36,3))
                    img = img.astype(np.float32)

                    #get prediction
                    interpreter_malaria.set_tensor(input_details[0]['index'], img)
                    interpreter_malaria.invoke()
                    output_data = interpreter_malaria.get_tensor(output_details[0]['index'])
                    pred = np.argmax(output_data[0])

                    return {'output': 'Sucess', 'msg': str(pred)}, 200
                else:
                    print('No Image Found')
                    return {'output': 'Error', 'msg': 'No Image Data Found'}, 200
            elif JSON.get('image', None) == "pneumonia":
                if JSON.get('imageData', None) != None:
                    data_url = JSON['imageData'].split(",")[1]
                    img_bytes = base64.b64decode(data_url, ' /')
                    img = Image.open(BytesIO(img_bytes))
                    img = img.resize((36, 36))
                    img = np.asarray(img)
                    img = img.reshape((1,36,36,1))
                    img = img.astype(np.float32)
                    img = img / 255.0
                    
                    #get prediction
                    interpreter_pneumonia.set_tensor(input_details[0]['index'], img)
                    interpreter_pneumonia.invoke()
                    output_data = interpreter_pneumonia.get_tensor(output_details[0]['index'])
                    pred = np.argmax(output_data[0])

                    return {'output': 'Sucess', 'msg': str(pred)}, 200
                else:
                    print('No Image Found')
                    return {'output': 'Error', 'msg': 'No Image Data Found'}, 200
            elif JSON.get('data', None) != None:
                to_predict_list = list(map(float, list(JSON.get('data').values())))

                #get prediction
                pred = predict(to_predict_list)
                
                if pred == -1:
                    return {'output': 'Error', 'msg': 'Wrong Data or Insufficient Dara'}, 200
                else:
                    return {'output': 'Sucess', 'msg': str(pred)}, 200
            else:
                return {'output': 'Error', 'msg': 'No Data or Wrong Data'}
        except Exception as ex:
            print (ex)
            return {'output': 'Error', 'msg': 'Unknown Error'}

if __name__ == '__main__':
	app.run(debug = True)