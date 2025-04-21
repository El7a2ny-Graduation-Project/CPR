from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
model.export(format="onnx", opset=13)  # Force opset=13 instead of default 17/18

#~#############################################################################################################
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("yolo11n-pose.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("yolo11n-pose_tf")

#~#############################################################################################################
import tensorflow as tf
    
converter = tf.lite.TFLiteConverter.from_saved_model("yolo11n-pose_tf")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,      # TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS         # Enable Flex (TF fallback) ops
]
tflite_model = converter.convert()

with open("yolo11n-pose_flex.tflite", "wb") as f:
    f.write(tflite_model)