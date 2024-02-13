import os
model_input_shape = (384,640)
# pt_model_path="test_models/best.pt"
pt_model_path='runs/train/exp13/weights/best.pt'
onnx_model_path=pt_model_path.split('.')[0]+".onnx"
trt_model_path=pt_model_path.split('.')[0]+".engine"
model_dir = '/'.join(list(pt_model_path.split('/')[:-1])) +'/'

output_shape_for_dynamic = (1,7,8400)

print("*"*50)
print("onnx path:", onnx_model_path)
print("trt path:", trt_model_path)
print("model dir:", model_dir)
print("-"*50)

"""using ultralytics model.export"""
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
model = YOLO(pt_model_path)  # load a pretrained model (recommended for training)
model.model.cuda().half()

path = model.export(format="onnx", dynamic=True,  simplify=True, imgsz=model_input_shape)#, half=True)  # export the model to ONNX format
os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_ultrlytics_export_dynamic_fp32.onnx")

path = model.export(format="onnx", dynamic=False,  simplify=True, imgsz=model_input_shape)#, half=True)  # export the model to ONNX format
os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_ultrlytics_export_static_fp32.onnx")

model.model.cuda()
path = model.export(format="engine", dynamic=True,  simplify=True, device=0, imgsz=model_input_shape)#, half=True)  # export the model to ONNX format
os.system("mv "+trt_model_path + " " + trt_model_path.split('.')[0] + "_ultrlytics_export_pt2trt_dynamic_fp32.engine")

path = model.export(format="engine", dynamic=False,  simplify=True, device=0, imgsz=model_input_shape)#, half=True)  # export the model to ONNX format
os.system("mv "+trt_model_path + " " + trt_model_path.split('.')[0] + "_ultrlytics_export_pt2trt_static_fp32.engine")

"""using torch export for pt2onnx convertion"""
import torch
import torch.nn
import cv2
import time 
import onnx
import onnxruntime
import numpy as np

from ultralytics.nn.tasks import attempt_load_weights, attempt_load_one_weight

model = attempt_load_weights(pt_model_path,
                             device=torch.device('cuda'),
                             inplace=True,
                             fuse=True)
#input_tensor = torch.ones((1,3,640,640)).cuda()
input_tensor = torch.ones((1,3,*model_input_shape)).cuda()
# static fp32
with torch.no_grad():
    print(f'process model:{pt_model_path}...')
    torch.onnx.export(model,
            input_tensor,
            onnx_model_path,
            opset_version=11,
            input_names=['images'],
            output_names=['output0'],
            dynamic_axes=None)
    onnx_model = onnx.load(onnx_model_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print('model incorrect')
        print(e)
    else:
        os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_torch_export_static_fp32.onnx")
        print('model correct')

## dynamic fp32
##dynamic_axes ={'input':{0:'batch',2:'H',3:'W'},
##dynamic_axes ={'input':{2:'H',3:'W'},
#dynamic_axes ={'images':{2:'H', 3:'W'},
#        #'output0':{2:'H',3:'W'},
#        #'output1':{2:'H',3:'W'},
#}
#with torch.no_grad():
#    print(f'process model:{pt_model_path}...')
#    torch.onnx.export(model,
#            input_tensor,
#            onnx_model_path,
#            opset_version=11,
#            input_names=['images'],
#            output_names=['output0'],
#            dynamic_axes=dynamic_axes)
#    onnx_model = onnx.load(onnx_model_path)
#    try:
#        onnx.checker.check_model(onnx_model)
#    except Exception as e:
#        print('model incorrect')
#        print(e)
#    else:
#        os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_torch_export_dynamic_fp32.onnx")
#        print('model correct')
#
"""using trt api for onnx2trt convertion"""
import tensorrt as trt
import os
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()
def get_engine(onnx_file_path, engine_file_path="", fp16=False, dynamic_in=False, dynamic_out=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(onnx_file_path, engine_file_path, fp16=False, dynamic_in=False, dynamic_out=False):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 32  # 4GB
            if fp16:
                assert (builder.platform_has_fast_fp16 == True), "not support fp16"
                config.flags = 1<<int(trt.BuilderFlag.FP16)
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # network.get_input(0).shape = [1, 3, 608, 608]

            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))

            if dynamic_in or dynamic_out:
                print("dynamic_in or out:", dynamic_in, dynamic_out)
                # Dynamic input setting 动态输入在builder里面设置
                profile = builder.create_optimization_profile()
                #最小的尺寸,常用的尺寸,最大的尺寸,推理时候输入需要在这个范围内
                profile.set_shape('images',(1,3,1,model_input_shape[1]),\
                        (1,3,model_input_shape[0]*3//4,model_input_shape[1]),(1,3,*model_input_shape))

#                profile.set_shape('images',(3,1,model_input_shape[1]),\
#                        (3,model_input_shape[0]*3//4,model_input_shape[1]),(3,*model_input_shape))
#                profile.set_shape('output0', output_shape_for_dynamic)
                config.add_optimization_profile(profile)

            plan = builder.build_serialized_network(network, config)
            print('plan:', plan, network, config, flush=True)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine
    return build_engine(onnx_file_path, engine_file_path,\
            fp16=fp16, dynamic_in=dynamic_in, dynamic_out=dynamic_out)

for modelname in [os.path.join(model_dir, item) for item in os.listdir(model_dir)]:
#for modelname in [model_dir+"best_ultrlytics_export_static_fp32.onnx"]:
#for modelname in [model_dir+"best_ultrlytics_export_dynamic_fp32.onnx"]:
    if not modelname.endswith('.onnx'):
        continue
    bare_name = modelname.split('.')[0]
    engine_name = bare_name + '.engine'
    print('-'*50)
    print("src modelname:", modelname)
    print('dst engine name:', engine_name)
    dynamic_in = False
    if 'dynamic' in bare_name.split('/')[-1]:
        dynamic_in = True
    dynamic_out = False
    try:
        # static fp32
        print('static fp32:')
        if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_static_fp32.engine"):
            print(engine_name.split('.')[0] + "_onnx_trtapi_static_fp32.engine exists.")
            assert 0
        get_engine(modelname, engine_name, fp16=False, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
        os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_static_fp32.engine")
        print(modelname + " static fp32 convert success")
    except:
        print(modelname + " static fp32 convert failed")

    print('-'*50)
    try:
        # dynamic fp32
        print('dynamic fp32:')
        if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp32.engine"):
            print(engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp32.engine exists")
            assert 0
        dynamic_out = True
        get_engine(modelname, engine_name, fp16=False, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
        os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp32.engine")
        print(modelname + " dynamic fp32 convert success")
    except:
        print(modelname + " dynamic fp32 convert failed")

    print('-'*50)
    try:
        # static fp16
        print('static fp16:')
        if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine"):
            print(engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine exists")
            assert 0
        get_engine(modelname, engine_name, fp16=True, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
        os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine")
        print(modelname + " static fp16 convert success")
    except:
        print(modelname + " static fp16 convert failed")

    print('-'*50)
    try:
        # dynamic fp16
        print('dynamic fp16:')
        dynamic_out = True
        if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp16.engine"):
            print(engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp16.engine exists")
            assert 0
        get_engine(modelname, engine_name, fp16=True, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
        os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp16.engine")
        print(modelname + " dynamic fp16 convert success")
    except:
        print(modelname + " dynamic fp16 convert failed")
    print('-'*50)
