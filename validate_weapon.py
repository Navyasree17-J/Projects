import os

# FORCE OFFLINE MODE
os.environ["ULTRALYTICS_OFFLINE"] = "1"

from ultralytics import YOLO

model_path = r"D:\PAT_Notes\Projects\Crowd_Anomaly_Project\models\weapon_best.pt"
data_yaml  = r"D:\PAT_Notes\Projects\Crowd_Anomaly_Project\weapon_test.yaml"

# DEBUG CHECK (VERY IMPORTANT)
print("Model exists:", os.path.exists(model_path))
print("YAML exists :", os.path.exists(data_yaml))

model = YOLO(model_path)
results = model.val(data=data_yaml)

print("✅ Validation finished")
