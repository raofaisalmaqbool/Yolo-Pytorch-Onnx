#!/usr/bin/env python3
"""
YOLO Inference Script
Uses yolo11n.pt model to run inference on image.jpg
Saves annotated results and prints detection details
"""

from ultralytics import YOLO
import os

def main():
    # Paths
    model_path = "model/yolo11n.pt"
    image_path = "static/image.jpg"
    output_path = "static/output.jpg"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    print(f"Loading YOLO model from {model_path}...")
    # Load the YOLO model
    model = YOLO(model_path)
    
    print(f"Running inference on {image_path}...")
    # Run inference
    results = model(image_path)
    
    # Process results
    for i, result in enumerate(results):
        print(f"\n{'='*60}")
        print(f"Image: {image_path}")
        print(f"{'='*60}")
        
        # Get detection data
        boxes = result.boxes
        
        if len(boxes) == 0:
            print("No detections found in the image.")
        else:
            print(f"\nFound {len(boxes)} detection(s):\n")
            
            # Print detection details
            for j, box in enumerate(boxes, 1):
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence score
                confidence = box.conf[0].cpu().numpy()
                
                # Get class ID
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = model.names[class_id]
                
                print(f"Detection {j}:")
                print(f"  Class ID: {class_id}")
                print(f"  Class Name: {class_name}")
                print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
                print(f"  Bounding Box:")
                print(f"    Top-Left: ({x1:.2f}, {y1:.2f})")
                print(f"    Bottom-Right: ({x2:.2f}, {y2:.2f})")
                print(f"    Width: {x2-x1:.2f}, Height: {y2-y1:.2f}")
                print()
        
        # Save annotated image
        annotated_image = result.plot()
        result.save(filename=output_path)
        print(f"{'='*60}")
        print(f"Annotated image saved to: {output_path}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

