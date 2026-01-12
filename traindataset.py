import yaml
from ultralytics import YOLO

def main():
    # Load configuration file for dataset paths and class names
    with open(r'Poachers Detection.v1i.yolov9\data.yaml', 'r') as file:
        data_config = yaml.safe_load(file)

    # Print out the dataset configuration
    print("Data Config Loaded:", data_config)

    # Initialize the YOLO model (pretrained YOLOv8 nano)
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data=r'Poachers Detection.v1i.yolov9\data.yaml',  # Dataset config
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,          # Safe now because of __main__ guard
        device=0            # GPU (RTX 3050). Use 'cpu' if needed
    )

    # Export trained model
    model.export()

    print("Training completed successfully.")

# 🔴 REQUIRED on Windows for multiprocessing
if __name__ == "__main__":
    main()
