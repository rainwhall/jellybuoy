import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("best.pt")

#List and DataFrame setup
data = [[], [], [], [], [], [], []]
columns = ['name', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2']

#Detection mainflow
try:
    while True:
        results = model(source=0, show=True, stream=True)

        for result in results:
            boxes = result.boxes
            names = result.names
            
            for box in boxes:
                #Getting values
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = result.names[int(cls)]

                #Append data into the list(s)-in-list
                data[0].append(name)
                data[1].append(cls)
                data[2].append(conf)
                data[3].append(xyxy[0])
                data[4].append(xyxy[1])
                data[5].append(xyxy[2])
                data[6].append(xyxy[3])

#Log data into DataFrame at KeyboardInterrupt
except KeyboardInterrupt:
    print("Logged successfully.")
    df = pd.DataFrame(data)
    df.insert(0, 'Values', columns)
    print(df)

    #Convert DataFrame to a CSV file
    df.to_csv(f'detections.csv')

finally:
    print("Exited program.")