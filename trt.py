import pycuda.driver as cuda
import time
import tensorrt as trt
import cv2
def _load_engine(engine_file_path):
    trt_logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_file_path, 'rb') as f:
        with trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            print('_load_engine ok.')
    return engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def _allocate_buffer(engine):
    binding_names = []
    for idx in range(100):
        bn = engine.get_binding_name(idx)
        if bn:
            binding_names.append(bn)
        else:
            break

    inputs = []
    outputs = []
    bindings = [None] * len(binding_names)
    stream = cuda.Stream()

    for binding in binding_names:
        binding_idx = engine[binding]
        if binding_idx == -1:
            print("Error Binding Names!")
            continue

        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings[binding_idx] = int(device_mem)

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def _test_engine(engine_file_path, data_input, num_times=100):
    # Code from blog.csdn.net/TracelessLe
    engine = _load_engine(engine_file_path)
    # print(engine)
    input_bufs, output_bufs, bindings, stream = _allocate_buffer(engine)
    batch_size = 8
    context = engine.create_execution_context()
    ###heat###
    input_bufs[0].host = data_input
    cuda.memcpy_htod_async(
        input_bufs[0].device,
        input_bufs[0].host,
        stream
    )
    context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle
    )
    cuda.memcpy_dtoh_async(
        output_bufs[0].host,
        output_bufs[0].device,
        stream
    )
    stream.synchronize()
    trt_outputs = [output_bufs[0].host.copy()]
    ##########
    start = time.time()
    for _ in range(num_times):
        time_bs1 = time.time()
        input_bufs[0].host = data_input
        cuda.memcpy_htod_async(
            input_bufs[0].device,
            input_bufs[0].host,
            stream
        )
        context.execute_async_v2(
            bindings=bindings,
            stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(
            output_bufs[0].host,
            output_bufs[0].device,
            stream
        )
        stream.synchronize()
        trt_outputs = [output_bufs[0].host.copy()]
        time_bs2 = time.time()
        time_use_bs = time_bs2 - time_bs1
        print(f'TRT use time {time_use_bs} for bs8')

    end = time.time()
    time_use_trt = end - start
    print(f"TRT use time {(time_use_trt)}for loop {num_times}, FPS={num_times * batch_size // time_use_trt}")
    return trt_outputs


def test_engine(data_input, loop=1):
    engine_file_path = r'D:\code\CrowdCounting-P2PNet\ckpt\best_mae.engine'
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()
    trt_outputs = None
    try:
        trt_outputs = _test_engine(engine_file_path, data_input, loop)
    finally:
        cuda_ctx.pop()
    return trt_outputs

if __name__ == '__main__':
    i = cv2.imread(r'D:\code\CrowdCounting-P2PNet\P2Pnet_dataset\test\0118_134336_025.bmp')
    #resize
    i = cv2.resize(i, (1280, 1280))
    a = test_engine(i)