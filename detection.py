from ultralytics import YOLO

# Load a model
model = YOLO("yolo26l.pt")  # load pretrained model
#model = YOLO("runs/detect/train-9/weights/best.pt")  # load pretrained model

results = model.train(data="data_det_light100_dark100.yaml", epochs=300, rect=True, patience=0) #imgsz=640)
