import os
import warnings
import onnx
import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder



class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, classes = torch.max(x[:, :, 4:], 2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

def yolov8_export(weights, device):
    model = YOLO(weights)
    model = deepcopy(model.model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False
            m.export = True
            m.format = 'onnx'
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model

def onnx_export(weights,
         output_dir,
         output_name,
         dynamic: bool = True, 
         simplify: bool = False,
         size: int = 640,
         batch: int =1,
         opset: int =16,
         ):
    suppress_warnings()
    print('Opening YOLOv8 model\n')

    device = select_device('cpu')
    model = yolov8_export(weights, device)

    if len(model.names.keys()) > 0:
        print('\nCreating labels.txt file')
        f = open('labels.txt', 'w')
        for name in model.names.values():
            f.write(name + '\n')
        f.close()

    model = nn.Sequential(model, DeepStreamOutput())

    if isinstance(size, int):
        img_size = (size, size)
    elif isinstance(size, list) and all(isinstance(i, int) for i in size):
        img_size = tuple(size)
    else:
        raise ValueError("size must be an integer or a list of integers")

    onnx_input_im = torch.zeros(batch, 3, *img_size).to(device)
    onnx_output_file = os.path.join(output_dir,f"{output_name}.onnx")

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'boxes': {
            0: 'batch'
        },
        'scores': {
            0: 'batch'
        },
        'classes': {
            0: 'batch'
        }
    }

    print('\nExporting the model to ONNX')
    torch.onnx.export(model, onnx_input_im, onnx_output_file, verbose=False, opset_version=opset,
                      do_constant_folding=True, input_names=['input'], output_names=['boxes', 'scores', 'classes'],
                      dynamic_axes=dynamic_axes if dynamic else None)

    if simplify:
        print('Simplifying the ONNX model')
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print('Done: %s\n' % onnx_output_file)


