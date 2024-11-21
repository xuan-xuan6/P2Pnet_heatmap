import cv2
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
# 初始化(创建引擎，为输入输出开辟&分配显存/内存.)
def init():
    model_path = r"D:\code\CrowdCounting-P2PNet\ckpt\latest.engine"
    # 加载runtime，记录log
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    # 反序列化模型
    engine = runtime.deserialize_cuda_engine(open(model_path, "rb").read())
    # print("输入",engine.get_binding_shape(0))
    # print("输出",engine.get_binding_shape(1))
    # 1. Allocate some host and device buffers for inputs and outputs:
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    # 推理上下文
    context = engine.create_execution_context()
    return context, h_input, h_output, stream, d_input, d_output
# 加载数据并将其喂入提供的pagelocked_buffer中.
def load_normalized_data(data_path, pagelocked_buffer, target_size=(640, 640)):
    img = cv2.imread(data_path)
    img = cv2.resize(img, target_size, cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 127.5
    img -= 1.
    img = np.transpose(np.array([img], dtype="float32"), (0, 3, 1, 2))
    # 此时img.shape为H * W * C: 224, 224, 3
    # print("图片shape", img.shape)
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, img.ravel())

# 推理
def inference(data_path):
    global context, h_input, h_output, stream, d_input, d_output
    load_normalized_data(data_path, h_input)
    t1 = time.time()
    # 将图片数据送到cuda显存中
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # 模型预测
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # 将结果送回内存中
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    ## 异步等待结果
    stream.synchronize()
    # Return the host output.
    print("推理时间", time.time() - t1)
    return h_output


if __name__ == '__main__':
    context, h_input, h_output, stream, d_input, d_output = init()
    img_path = r"D:\code\CrowdCounting-P2PNet\P2Pnet_dataset\test\0118_134336_025.bmp"
    # for image in range(10):
    output = inference(data_path=img_path)
    print("type  output:", type(output), "output.shape:", output.shape, "output:", output, "\n")