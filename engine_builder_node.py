import os
import folder_paths
from .export_trt import export_trt
import urllib.request
import shutil

ENGINE_DIR = os.path.join(folder_paths.models_dir, "tensorrt", "yolo-nas-pose")
ONNX_DIR = os.path.join(folder_paths.models_dir, "onnx", "yolo-nas-pose")

# Model URLs and configurations
YOLO_NAS_POSE_MODELS = {
    "large_0.1": {
        "url": "https://huggingface.co/yuvraj108c/yolo-nas-pose-onnx/resolve/main/yolo_nas_pose_l_0.1.onnx",
        "filename": "yolo_nas_pose_l_0.1.onnx",
        "engine_name": "yolo_nas_pose_l_0.1-fp16.engine"
    },
    "large_0.2": {
        "url": "https://huggingface.co/yuvraj108c/yolo-nas-pose-onnx/resolve/main/yolo_nas_pose_l_0.2.onnx",
        "filename": "yolo_nas_pose_l_0.2.onnx",
        "engine_name": "yolo_nas_pose_l_0.2-fp16.engine"
    },
    "large_0.35": {
        "url": "https://huggingface.co/yuvraj108c/yolo-nas-pose-onnx/resolve/main/yolo_nas_pose_l_0.35.onnx",
        "filename": "yolo_nas_pose_l_0.35.onnx",
        "engine_name": "yolo_nas_pose_l_0.35-fp16.engine"
    },
    "large_0.5": {
        "url": "https://huggingface.co/yuvraj108c/yolo-nas-pose-onnx/resolve/main/yolo_nas_pose_l_0.5.onnx",
        "filename": "yolo_nas_pose_l_0.5.onnx",
        "engine_name": "yolo_nas_pose_l_0.5-fp16.engine"
    },
    "large_0.8": {
        "url": "https://huggingface.co/yuvraj108c/yolo-nas-pose-onnx/resolve/main/yolo_nas_pose_l_0.8.onnx",
        "filename": "yolo_nas_pose_l_0.8.onnx",
        "engine_name": "yolo_nas_pose_l_0.8-fp16.engine"
    }
}

def download_onnx_model(model_key):
    """Download ONNX model if it doesn't exist"""
    if model_key not in YOLO_NAS_POSE_MODELS:
        return f"Invalid model key: {model_key}", None
    
    model_info = YOLO_NAS_POSE_MODELS[model_key]
    url = model_info["url"]
    filename = model_info["filename"]
    local_path = os.path.join(ONNX_DIR, filename)
    
    if os.path.exists(local_path):
        return f"Model already exists at: {local_path}", local_path
    
    try:
        print(f"Downloading model from {url} to {local_path}...")
        
        # Create a temporary file for downloading
        temp_file = local_path + ".tmp"
        
        # Download the file
        with urllib.request.urlopen(url) as response, open(temp_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        
        # Rename the temporary file to the target filename
        shutil.move(temp_file, local_path)
        
        return f"Successfully downloaded model to: {local_path}", local_path
    except Exception as e:
        return f"Error downloading model: {str(e)}", None

class YoloNasPoseEngineBuilder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_size": (list(YOLO_NAS_POSE_MODELS.keys()), {
                    "default": "large_0.5",
                    "tooltip": "Select the YoloNasPose model size. Larger models provide better accuracy but require more VRAM."
                }),
                "custom_engine_name": ("STRING", {
                    "default": "",
                    "tooltip": "Optional custom name for the TensorRT engine file. If empty, will use the default name based on the model."
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable FP16 precision for faster inference and lower VRAM usage. Disable if you experience stability issues."
                }),
                "custom_onnx_path": ("STRING", {
                    "default": "",
                    "optional": True,
                    "tooltip": "Optional path to a custom ONNX model file. If provided, will use this instead of downloading the predefined model."
                }),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "build_engine"
    CATEGORY = "tensorrt"
    OUTPUT_NODE = True
    
    def build_engine(self, model_size, custom_engine_name, use_fp16, custom_onnx_path=""):
        # Ensure directories exist
        os.makedirs(ENGINE_DIR, exist_ok=True)
        os.makedirs(ONNX_DIR, exist_ok=True)
        
        # Determine ONNX path and engine name
        if custom_onnx_path and os.path.exists(custom_onnx_path):
            onnx_path = custom_onnx_path
            engine_name = custom_engine_name if custom_engine_name else os.path.basename(custom_onnx_path).replace(".onnx", "-fp16.engine" if use_fp16 else ".engine")
        else:
            # Download and use predefined model
            model_info = YOLO_NAS_POSE_MODELS[model_size]
            message, onnx_path = download_onnx_model(model_size)
            
            if not onnx_path:
                return (message,)
                
            engine_name = custom_engine_name if custom_engine_name else model_info["engine_name"]
            if use_fp16 and not engine_name.endswith("-fp16.engine"):
                engine_name = engine_name.replace(".engine", "-fp16.engine")
            elif not use_fp16 and "-fp16.engine" in engine_name:
                engine_name = engine_name.replace("-fp16.engine", ".engine")
        
        engine_path = os.path.join(ENGINE_DIR, engine_name)
        
        # Check if engine already exists
        if os.path.exists(engine_path):
            return (f"Engine already exists at: {engine_path}. Delete it manually if you want to rebuild.",)
        
        try:
            print(f"Building TensorRT engine from {onnx_path} to {engine_path}...")
            result = export_trt(trt_path=engine_path, onnx_path=onnx_path, use_fp16=use_fp16)
            if result == 0:
                return (f"Successfully built engine: {engine_path}",)
            else:
                return (f"Failed to build engine. Check console for details.",)
        except Exception as e:
            return (f"Error building engine: {str(e)}",) 