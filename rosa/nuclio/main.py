import json
import base64
from PIL import Image
import io
import torch
import os
from ultralytics import YOLO

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    #model_name = 'best.pt'
    #model_path='/home/perevozchikovav/cvat/cvat/serverless/pytorch/ultralytics/astrantsia/nuclio/best.pt'
    #model = torch.hub.load(path=model_path, 'custom', source='local', force_reload=True)
    #model = torch.hub.load('ultralytics/yolov5', 'custom', source='local', path=model_name, force_reload = True)
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
    model = YOLO('yolov8x.pt')
    model = YOLO('./best.pt')
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run yolo-v8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    #yolo_results_json = context.user_data.model(image).boxes.xyxy[0].to_dict(orient='records')
    yolo_results = context.user_data.model(image)

    encoded_results = []
    for yolo_result in yolo_results:
        for result in yolo_result.boxes:
            encoded_results.append({
                'confidence': result.conf.item(),
                'label': yolo_results[0].names[0],
                'points': [
                    result.xyxy[0][0].item(),
                    result.xyxy[0][1].item(),
                    result.xyxy[0][2].item(),
                    result.xyxy[0][3].item()
                ],
                'type': 'rectangle'
            })

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)
