import ultralytics


# Choose the model name
model_name = 'yolo11m.pt'


# Load the model with pre-trained weights
print(f"Loading model: {model_name}")
model = ultralytics.YOLO(f'models/{model_name}')
print(f"Sucessfully loaded {model_name}")

# Train the model
results = model.train(data="configs/Face_Detection.yaml", epochs=10, batch=-1)