import torch

# Load model without AutoShape wrapper
model = torch.hub.load('ultralytics/yolov5', 'yolov5nu', pretrained=True)

# Export to ONNX
model.export(format='onnx')