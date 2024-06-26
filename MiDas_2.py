import cv2
import torch
import numpy as np
import time

# load midas model for depth estimation
# model_type = "DPT_Large"
# model_type = "DPT_Hybrid"
model_type = "MiDaS_small" #MiDas v2.1 - small (lowest accuracy, highest infernec speed)
midas = torch.hub.load('intel-isl/MiDaS', model_type)

# move model to gpu if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
if  model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transforms = midas_transforms.dpt_transform
else:
    transforms = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    start = time.time()
    # Transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transforms(img).to(device)

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()
        print(output)
        output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime


    # plt.imshow(output)
    # cv2.imshow('CV2Frame', frame)
    # plt.pause(0.00001)

    cv2.putText(img, f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('image', img)
    cv2.imshow('Depth_map', output)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

# plt.show()