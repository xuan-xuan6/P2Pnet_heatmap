import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2  # 如果你需要对图像进行预处理

# 加载 TensorRT 引擎
def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 图像预处理函数 (根据P2PNet的要求调整)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1280, 1280))  # 假设P2PNet输入为640x640
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # 将通道放到前面
    img = np.expand_dims(img, axis=0)  # 增加batch维度
    return img

# 推理函数
def infer(engine, input_data):
    context = engine.create_execution_context()

    # 获取输入输出的绑定索引
    input_index = engine.get_binding_index("input")  # 假设模型的输入名称为 "input"
    output_index = engine.get_binding_index("output")  # 假设模型的输出名称为 "output"

    # 获取输入输出形状
    input_shape = engine.get_binding_shape(input_index)
    output_shape = engine.get_binding_shape(output_index)

    # 确保输入数据形状匹配
    assert input_data.shape == tuple(input_shape), "输入数据的形状与引擎期望的输入形状不匹配"

    # 分配 GPU 缓存
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(trt.volume(output_shape) * input_data.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]

    # 创建流
    stream = cuda.Stream()

    # 将输入数据拷贝到 GPU
    cuda.memcpy_htod_async(d_input, input_data, stream)

    # 执行推理
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # 分配输出数据的空间
    output = np.empty(output_shape, dtype=np.float32)

    # 将输出数据拷贝回 CPU
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    return output

# 使用示例
if __name__ == "__main__":
    engine_file_path = r"D:\code\CrowdCounting-P2PNet\ckpt\best_mae.engine"  # 替换为你的 engine 文件路径
    image_path = r"D:\code\CrowdCounting-P2PNet\P2Pnet_dataset\test\0118_134336_025.bmp"  # 替换为你的图像路径

    # 图像预处理
    input_data = preprocess_image(image_path)

    # 加载模型
    engine = load_engine(engine_file_path)

    # 推理
    output_data = infer(engine, input_data)

    print("Inference output:", output_data)

    # 如果输出是密度图，可以显示或处理
    if output_data.ndim == 4:  # 可能是 (batch, channel, height, width)
        density_map = output_data[0, 0, :, :]  # 假设输出的是密度图
        cv2.imshow('Density Map', density_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Predicted count: {output_data}")