from ultralytics import YOLO

# Load a model
model = YOLO("yolov5n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=1)  # train the model
