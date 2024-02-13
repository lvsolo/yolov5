import tensorrt as trt
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
from ultralytics.utils import ops 
import numpy as np
import cv2
import tqdm
from utils.augmentations import letterbox

model_dir = 'runs/train/exp13/weights/'

test_image_path = '/media/lvsolo/CA89-5817/datasets/helmat/bdd100k_images/bdd100k/yolov5/images/val/b8f0315c-0fb7602c.jpg'
# test_image_path = 'data/images/zidane.jpg'
letterbox_shape = (384, 640)
letterbox_auto = False
normalize_std = 1.#255.
if_rgb2bgr = True
input_np_image_type = np.float32
output_shape_for_dynamic_engine = (1,7,8400)

test_infer_times = 1

def if_dynamic_engine(path):
#    if 'dynamic_fp32.engine' in path.split('/')[-1] or \
#        'dynamic_fp16.engine' in path.split('/')[-1]:
    if 'dynamic' in path.split('/')[-1]:
        return True

def plot_with_xyxys(xyxys, img_path, out_dir=None):
    objs = xyxys
    img = cv2.imread(img_path)
    for obj in objs:
        img = cv2.rectangle(img, \
            ( int(obj[0]), int(obj[1])), \
            ( int(obj[2]), int(obj[3])),\
            color=(0,0,255), thickness=2)
#print(img_path.split('/')[-1])
    if not out_dir:
        if not os.path.exists(img_path):
            cv2.imwrite(img_path.split('/')[-1], img)
        else:
            cv2.imwrite('new_'+img_path.split('/')[-1], img)
    else:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_dir + '/' + img_path.split('/')[-1], img)


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def run(path, dynamic_engine=False):
    #path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.fp16_static_input"#
#    path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.dynamicinput"#
    #path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine"#.dynamicinput"#
    is_dynamic_engine = if_dynamic_engine(path)
    out_dir_path = path.split('/')[-1].replace('.','_') + '/'

    engine = load_engine(path)
    imgpath = test_image_path
    #imgpath = 'R-C.jpg'
    
    context = engine.create_execution_context()
    image1 = cv2.imread(imgpath)
    
    total_st = time.time()
    st = total_st
    image = letterbox(image1, letterbox_shape, stride=32, auto=letterbox_auto)[0]
    if normalize_std:
        image = image / normalize_std
    if if_rgb2bgr:
        image = image[:,:,::-1]
    image = image.transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    image = np.ascontiguousarray(image)
    image = np.expand_dims(image, axis=0)
    
    # tensorrt8.6.1即使是fp16的engine也不能直接使用fp16输入，8.5版本貌似可以
    image = image.astype(input_np_image_type)
    print('input image type:', image.dtype)
    print('input image shape:', image.shape)
    
    outshape= context.get_binding_shape(1) 
    """动态输入通过get_binding_shape(0|1)时获得的值可能是0或者包含-1,此时应指定输入输出的大小为实际大小,
    并通过set_binding_shape方式进行指定
    """
#    if is_dynamic_engine:
#        outshape= output_shape_for_dynamic_engine 
    print('outshape:',outshape, flush=True)
    output = np.empty((outshape), dtype=np.float32)

    org_inshape = context.get_binding_shape(0)
    org_outshape = context.get_binding_shape(1) 
    print('inshape from get_binding_shape(0):', org_inshape)
    print('outshape from get_binding_shape(1):', org_outshape)
    # for i in range(5):
    #     print('ind: {}, {}'.format(i, context.get_binding_shape(i)))

    print('dynamic inshape:', image.shape)
    print('dynamic outshape:', output.shape)
    if is_dynamic_engine or dynamic_engine:
        """动态输入通过指定输入输出的大小为实际大小
        """
        st_set_binding = time.time()
        context.set_binding_shape(0, image.shape)
#        context.set_binding_shape(1, output.shape)
        print('set binding time:', time.time() - st_set_binding)
    
    d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    
    cuda.memcpy_htod(d_input,image)
    print('preprocess time:',time.time()- st)
    st = time.time()
    for _ in range(test_infer_times):
        context.execute_v2(bindings)
    print('test avg infer time :', (time.time()- st)/test_infer_times)
    st = time.time()
    cuda.memcpy_dtoh(output, d_output)
    import torch
    pred = output
    print('pred shape', pred[0].shape, flush=True)
    # pred[0] = pred[0].reshape(3,-1,15)
    # pred[0] = pred[0][:,:48*80,:]
    # pred[0] = torch.squeeze(pred[0], 0)
    # pred = np.transpose(pred, (0,2,1))

    
    tmp = pred[0].reshape(3,-1,15)
    # tmp = tmp[:,:48*80,:]
    tmp = tmp[:,48*80:48*80+24*40,:]
    pred = np.transpose(tmp, (0,2,1))
    for i in range(len(pred[0])):
        print('gggggggggggggggggg pred shape', pred[0][i], flush=True)

    #print('pred shape', np.max(pred[0], axis=0), flush=True)
    #print('pred shape', np.min(pred[0], axis=0), flush=True)
    pred = torch.from_numpy(pred).cuda()
    #TODO 如何通过内存地址、数据类型和数据size来init一个torch.cuda.tensor
    #pred = torch.cuda.FloatTensor(d_output, output.size)
    pred = ops.non_max_suppression(pred,
                                   conf_thres=0.5,#self.args.conf,
                                   iou_thres=0.5,#self.args.iou,
                                   classes = None,
#                                   classes=['person','rider','car','bus','truck','bike','motor','traffic light','traffic sign','train'],
                                   agnostic=True,#agnostic=False,#self.args.agnostic_nms,
                                   max_det=300,#self.args.max_det,
                                #    nc=10
                                   )#classes=None)#self.args.classes)[0]
    print('postprocess time:',time.time()- st)
    print('pred shape', pred[0].shape, flush=True)
#    pred[0] = pred[0].reshape(-1,6)
#    print('pred shape', pred[0].shape, flush=True)
##    for i in range(100):
##        print('pred shape', pred[0][i], flush=True)
#    print('pred shape', np.max(pred[0].cpu().numpy(), axis=1), flush=True)
#    print('pred shape', np.min(pred[0].cpu().numpy(), axis=1), flush=True)
#    for i in range(len(pred[0])):
#        print('pred shape', pred[0][i], flush=True)

    #visualization
    xyxys = []
    confs = []
    cls = []
    for i, det in enumerate(pred):
        print('before scalebox:', det)
        print(image.shape[2:], det[:,:4], image1.shape)
        det[:,:4] = ops.scale_boxes(image.shape[2:], det[:,:4], image1.shape).round()
        print('after scalebox:', det)
        xyxys.append(det[:,:4].cpu().numpy().tolist())
        confs.append(det[:,4].cpu().numpy().tolist())
#        cls.append(np.argmax(det[:,5:].cpu().numpy().astype(np.int32)-5, axis=1).tolist())
        cls.append(det[:,5].cpu().numpy().astype(np.int32).tolist())
        print("dets.shape dets:", det.shape, det[:6].cpu().numpy())
    xyxys = xyxys[0]
    confs = confs[0]
    cls = cls[0]
    print("xyxys confs and cls:", xyxys,confs, cls)
    plot_with_xyxys(xyxys, imgpath, out_dir=out_dir_path)
    
    print('total time:', time.time()- total_st)
    print(out_dir_path, flush=True)

#engines_path_list = [
#    'test_models/best_torch_export_static_fp32_onnx_trtapi_dynamic_fp16.engine'
#    ,'test_models/best_torch_export_static_fp32_onnx_trtapi_dynamic_fp32.engine'
#    ,'test_models/best_torch_export_static_fp32_onnx_trtapi_static_fp16.engine'
#    ,'test_models/best_torch_export_static_fp32_onnx_trtapi_static_fp32.engine'
#    ,'test_models/best_ultrlytics_export_dynamic_fp32_onnx_trtapi_dynamic_fp16.engine'
#    ,'test_models/best_ultrlytics_export_dynamic_fp32_onnx_trtapi_dynamic_fp32.engine'
#    ,'test_models/best_ultrlytics_export_dynamic_fp32_onnx_trtapi_static_fp16.engine'
#    ,'test_models/best_ultrlytics_export_dynamic_fp32_onnx_trtapi_static_fp32.engine'
#    ,'test_models/best_ultrlytics_export_static_fp32_onnx_trtapi_static_fp32.engine'
#]
engines_path_list = [
#    'test_models/best_ultrlytics_export_dynamic_fp32_onnx_trtapi_static_fp32.engine'
#   ,'test_models/best_ultrlytics_export_dynamic_fp32_onnx_trtapi_dynamic_fp32.engine'
#   ,'test_models/best_ultrlytics_export_dynamic_fp32_onnx_trtapi_static_fp16.engine'
#   ,'test_models/best_ultrlytics_export_dynamic_fp32_onnx_trtapi_dynamic_fp16.engine'
##   'test_models/best_torch_export_static_fp32_onnx_trtapi_static_fp32.engine'
#   ,'test_models/best_ultrlytics_export_static_fp32_onnx_trtapi_static_fp32.engine'
#   'test_models/best_ultrlytics_export_pt2trt_dynamic_fp32.engine'
#   ,'test_models/best_ultrlytics_export_pt2trt_static_fp32.engine'
#'/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/runs/train/exp13/weights/best_onnx_trtapi_static_fp32.engine',
#'/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/runs/train/exp13/weights/best_ultrlytics_export_pt2trt_dynamic_fp32.engine',
#'/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/runs/train/exp13/weights/best_ultrlytics_export_pt2trt_static_fp32.engine',
#'/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/runs/train/exp13/weights/best_ultrlytics_export_static_fp32_onnx_trtapi_static_fp32.engine'
#'/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/runs/train/exp13/weights/best_torch_export_static_fp32_onnx_trtapi_static_fp32.engine'
'/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/runs/train/exp28/weights/best.engine'
# '/home/lvsolo/miniconda3/envs/helmet_yolov5/lib/python3.7/site-packages/cvu/detector/yolov5/backends/weights/yolov5s_384_640_fp16_trt.engine'

]  
   
for path in engines_path_list:
#for path in [os.path.join(model_dir, item) for item in os.listdir(model_dir)]:                                   
    if not path.endswith('.engine'):
        continue
    print(path)
    run(path)
##    try:
##        run(path)
##    except:
##        print("ERROR "+ path)

#import torch
#from collections import *
#for path in engines_path_list:
#    # 判断版本
#    # check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
#    device = torch.device('cuda:0')
#    # 1.创建一个Binding对象，该对象包含'name', 'dtype', 'shape', 'data', 'ptr'这些属性
#    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
#    logger = trt.Logger(trt.Logger.INFO)
#    # 2.读取engine文件并记录log
#    with open(path, 'rb') as f, trt.Runtime(logger) as runtime:
#        # 将engine进行反序列化，这里的model就是反序列化中的model
#        model = runtime.deserialize_cuda_engine(f.read())  # model <class 'tensorrt.tensorrt.ICudaEngine'> num_bindings=2,num_layers=163
#    # 3.构建可执行的context(上下文：记录执行任务所需要的相关信息)
#    context = model.create_execution_context()  # <IExecutionContext>
#    bindings = OrderedDict()
#    output_names = []
#    fp16 = False  # default updated below
#    dynamic = False
#    print(model.num_bindings)
#    for i in range(model.num_bindings):
#        name = model.get_binding_name(i) # 获得输入输出的名字"images","output0"
#        print(model.num_bindings, name)
#        dtype = trt.nptype(model.get_binding_dtype(i))
#        if model.binding_is_input(i):  # 判断是否为输入
#            if -1 in tuple(model.get_binding_shape(i)):  # dynamic get_binding_shape(0)->(1,3,640,640) get_binding_shape(1)->(1,25200,85)
#                dynamic = True
#                context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
#            if dtype == np.float16:
#                fp16 = True
#        else:  # output
#            output_names.append(name)  # 放入输出名字 output_names = ['output0']
#        shape = tuple(context.get_binding_shape(i))  # 记录输入输出shape
#        print(i, model.num_bindings, shape)
#        try:
#            im = torch.from_numpy(np.ones(shape, dtype=dtype)).to(device)  # 创建一个全0的与输入或输出shape相同的tensor
#        except:
#            print('1111111111111111111', flush=True)
#        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 放入之前创建的对象中
#    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())  # 提取name以及对应的Binding
#    batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
#
#    # print(binding_addrs, batch_size, output_names, bindings)
#
#    im1 = cv2.imread('data/images/zidane.jpg')
#    letterbox_shape = (384,640)
#    letterbox_auto = False
#    im = letterbox(im1, letterbox_shape, stride=32, auto=letterbox_auto)[0]
#    im = im.transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
#    im = np.ascontiguousarray(im)
#    im = np.expand_dims(im, axis=0)
#    im = torch.from_numpy(im).cuda()
#    s = bindings['images'].shape
#    assert im.shape == s, f"input size {im.shape} {'>' if dynamic else 'not equal to'} max model size {s}"
#    binding_addrs['images'] = int(im.data_ptr())
#    # 调用计算核心执行计算过程
#    context.execute_v2(list(binding_addrs.values()))
#    y = [bindings[x].data for x in sorted(output_names)]
#    # print(y)
#    for i in range(len(y)):
#        print(y[i].shape)