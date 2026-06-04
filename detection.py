from ultralytics import YOLO

data_files = [
    "data_light100_dark100.yaml",
    "data_light50_dark50.yaml",
    "data_light75_dark25.yaml",
    "data_light25_dark75.yaml",
    "data_light100_dark0.yaml",
    "data_light0_dark100.yaml",
    "data_light50_dark0.yaml",
    "data_light0_dark50.yaml",
    "data_light25_dark0.yaml",
    "data_light0_dark25.yaml"
]

for data_file in data_files:
    # Load a model
    model = YOLO("yolo26l.pt")  # load pretrained model
    #model = YOLO("runs/detect/train-9/weights/best.pt")  # load pretrained model
    results = model.train(data="data_det_light50_dark50.yaml", epochs=300, rect=True, patience=0) #imgsz=640)
