from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load pretrained model

results = model.train(data="data_det_light.yaml", epochs=100, imgsz=640)
