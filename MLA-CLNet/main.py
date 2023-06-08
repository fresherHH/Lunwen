import os
import io
import json

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, redirect, url_for
from flask_cors import CORS
from flask_wtf import FlaskForm
from wtforms import SubmitField

import os
import time
# from tkinter import image_names
# from cv2 import DrawMatchesFlags_DRAW_OVER_OUTIMG

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from Model.Last import Two_Stream_Netv1
import Model.DPNet.T1 as t1

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


app = Flask(__name__)
CORS(app)  # 解决跨域问题

# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

weights_path = "save_weights/detection/T1_Two_Stream_Net.pth"
assert os.path.exists(weights_path), "weights path does not exist..."
classification = t1.Two_Stream_Netv1()
classification = torch.load(weights_path, map_location=device)


weights_path_local = "save_weights/localization/best_model.pth"
assert os.path.exists(weights_path), "weights path does not exist..."
model = Two_Stream_Netv1(num_classes=2)
# load weights
model.load_state_dict(torch.load(weights_path_local, map_location='cpu')['model'])
model.to(device)

def classificatin():

    classification.to(device)
    classification.eval()
    return classification





def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)



def get_detection(image_bytes):
    model = classificatin()
    # load class info

    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = torch.softmax(model.forward(tensor).squeeze(), dim=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:

        return_info = {"result": [str(e)]}
    return return_info




@app.route("/detect", methods=["POST"])
@torch.no_grad()
def detect():
    image = request.files["file"]
    print(image)
    img_bytes = image.read()

    info = get_detection(image_bytes=img_bytes)
    return jsonify(info)

def show():
    import base64
    img_stream = ''
    img_local_path = r'F:\LunWen\static\mask.png'
    with open(img_local_path, 'r') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream


@app.route("/localize", methods=["POST"])
@torch.no_grad()
def localize():

    image = request.files["file"]

    image = Image.open(image)
    image_cv2 = np.array(image)



    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])


    img = data_transform(image_cv2)

    img = torch.unsqueeze(img, dim=0)


    model.eval()
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        # prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        # savepath = os.path.join(r"D:\SegHH\\mask\\", i.split(".")[0] + ".png")
        savepath = os.path.join(r"static\\", "mask.png")
        # mask.save("test_result.png")
        mask.save(savepath)

    info = "yes"
    return jsonify(info)






@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


@app.route("/detection", methods=["GET", "POST"])
def detection():
    return render_template("detection.html")


@app.route("/localization", methods=["GET", "POST"])
def localization():
    return render_template("localization.html")



@app.route("/show", methods=["GET", "POST"])
def show():
    import base64
    img_stream = ''
    img_local_path = r'F:\LunWen\static\mask.png'
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()

    return render_template("show.html", img_stream=img_stream)






if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




