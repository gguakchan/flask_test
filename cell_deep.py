from flask import Flask, request
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

import pymongo
from pymongo import MongoClient

model = torch.load('model.pt')

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체

class_names =('hela', 'ht', 'imr')

app = Flask(__name__)

@app.route('/receive_image', methods=['GET','POST'])
def receive_image():
    try:
        image = request.data
        image = transforms_test(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            print('result: ' + class_names[preds[0]])
    
            cluster = MongoClient("mongodb+srv://wonomedb:wonomedb123@wonome.3buk7.mongodb.net/")

            db = cluster["cell_classification"]
            collection = db["classification"]

            respond = class_names[preds[0]]
            post = {"result": respond}
            collection.insert_one(post)

        result = "Image received and analyzed."
        return result
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # 서버 시작