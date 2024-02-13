import torch
import torch.nn
import cv2
import time 
import onnx
import onnxruntime
import numpy as np

from ultralytics.nn.tasks import attempt_load_weights, attempt_load_one_weight

model_path='runs/train/exp8/weights/best.pt'
model = attempt_load_weights(model_path,
                               device=torch.device('cuda'),
                               inplace=True,
                               fuse=True)
#model=model.cuda()
model = model.to('cpu')
#model.load_state_dict(torch.load(model_path))
input_tensor = torch.ones((1,3,384,640))
# input_tensor = input_tensor.to('cuda')
model_name_tmp = model_path.split('.')[0].replace('/','_')

model_names = [model_name_tmp + '.onnx', \
    model_name_tmp+'_dynamic_batch.onnx', \
model_name_tmp + '_dynamic_hw.onnx']
#model_names = ["../runs/detect/train4/weights/best.onnx"]
dynamic_batch = {'images':{0:'batch'},
        'output0':{0:'batch'},
        #'output1':{0:'batch'},
        #'output2':{0:'batch'},
        #'output3':{0:'batch'},
        #'output4':{0:'batch'},
        #'output5':{0:'batch'},
        #'output6':{0:'batch'}
}
dynamic_hw ={'images':{0:'batch',2:'H',3:'W'},
        'output0':{0:'batch', 2:'H',3:'W'},
        #'output1':{2:'H',3:'W'},
        #'output2':{2:'H',3:'W'},
        #'output3':{2:'H',3:'W'},
        #'output4':{2:'H',3:'W'},
        #'output5':{2:'H',3:'W'},
        #'output6':{2:'H',3:'W'}
}
dynamic_=[None,dynamic_batch,dynamic_hw]


with torch.no_grad():
    for i,model_name in enumerate(model_names):
        print(f'process model:{model_name}...')
        torch.onnx.export(model,
                input_tensor,
                model_name,
                opset_version=11,
                input_names=['images'],
                output_names=['output0'],
#                output_names=['output0','output1','output2','output3'],
                dynamic_axes=dynamic_[i])

        print(f'onnx model:{model_name} saved successfully...')
        print(f'begin check onnx model:{model_name}...')

        onnx_model = onnx.load(model_name)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            print('model incorrect')
            print(e)
        else:
            print('model correct')

print('*'*50)
print('Begin to test...')
case_1 = np.random.rand(1,3,384,640).astype(np.float32)
case_2 = np.random.rand(2,3,384,640).astype(np.float32)
case_3 = np.random.rand(1,3,384,640).astype(np.float32)
cases = [case_1,case_2,case_3]
providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
for model_name in model_names:
    print('-'*50)
    onnx_session = onnxruntime.InferenceSession(model_name,providers=providers)
    for i,case in enumerate(cases):
        onnx_input = {'images':case}
        try:
            onnx_output = onnx_session.run(['output0'],onnx_input)
            print(len(onnx_output), onnx_output[0].shape)
            #onnx_output = onnx_session.run(['output0','output1','output2','output3'],onnx_input)[0]
        except Exception as e:
            print(f'Input:{i} on model:{model_name} failed')
            print(e)
        else:
            print(f'Input:{i} on model:{model_name} succeed')
