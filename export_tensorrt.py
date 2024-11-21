
import torch
from pathlib import Path
import tensorrt as trt
def export_to_engine(onnx_path, engine_path, input_size, batch, ch=3, input_shapes=None, fp16=False, dynamic=True,
                     workspace=4, device="cuda:0"):
    """
    将onnx文件转成engine
    Args:
        onnx_path: 加载的onnx文件路径
        engine_path: 保存engine文件路径
        input_size: 输入input尺寸
        batch: 批次大小
        ch：模型的通道数，默认为3；当ch=1时为单通道
        input_shapes: 输入动态batch的shape
        fp16: 是否使用FP16，无监督中默认False
        dynamic: 动态导出，模型shape[0]=-1
        workspace: 默认4
        device: 默认cuda

    Returns:

    """

    # 判断 onnx文件是否存在
    assert Path(onnx_path).exists(), f'failed to export ONNX file: {onnx_path}'

    torch.cuda.set_device(device)
    device = torch.device(device)

    logger = trt.Logger(trt.Logger.INFO)
    # 输出详细信息
    logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     workspace * 1 << 30)
    else:
        config.max_workspace_size = workspace * 1 << 30

    flag = (1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # onnx_model = onnx.load(onnx_path)
    # if not parser.parse(onnx_model.SerializeToString()):
    #     error_msgs = ''
    #     for error in range(parser.num_errors):
    #         error_msgs += f'{parser.get_error(error)}\n'
    #     raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'TensorRT: input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'TensorRT: output "{out.name}" with shape{out.shape} {out.dtype}')

    # im = torch.zeros(batch, 1, input_size, input_size).to(device)
    if isinstance(input_size, int):
        im = torch.zeros(batch, ch, input_size, input_size).to(device)
    elif isinstance(input_size, list) and len(input_size) == 2:
        im = torch.zeros(batch, ch, *input_size).to(device)

    if dynamic:
        shape = im.shape
        if shape[0] <= 1:
            print(f"TensorRT: WARNING ⚠️ 'dynamic=True' model requires max batch size, i.e. 'batch=16'")
        profile = builder.create_optimization_profile()

        if input_shapes is None:
            for inp in inputs:
                profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0] // 2), *shape[1:]), shape)
        else:
            for input_name, param in input_shapes.items():
                min_shape = param['min_shape']
                opt_shape = param['opt_shape']
                max_shape = param['max_shape']
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)

    print(
        f'TensorRT: building FP{16 if builder.platform_has_fast_fp16 and fp16 else 32} engine as {engine_path}')
    if builder.platform_has_fast_fp16 and fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Write file
    engine = builder.build_engine(network, config)
    with open(engine_path, mode="wb") as f:
        f.write(bytearray(engine.serialize()))

if __name__ == '__main__':
    # export_to_engine(onnx_path, engine_path, input_size, batch, ch=3, input_shapes=None, fp16=False, dynamic=True,
    #                  workspace=4, device="cuda"):
    """
    将onnx文件转成engine
    Args:
        onnx_path: 加载的onnx文件路径
        engine_path: 保存engine文件路径
        input_size: 输入input尺寸
        batch: 批次大小
        ch：模型的通道数，默认为3；当ch=1时为单通道
        input_shapes: 输入动态batch的shape
        fp16: 是否使用FP16，无监督中默认False
        dynamic: 动态导出，模型shape[0]=-1
        workspace: 默认4
        device: 默认cuda
    """
    onnx_path = r'D:\code\CrowdCounting-P2PNet\ckpt\latest_test.onnx'
    engine_path = r'D:\code\CrowdCounting-P2PNet\ckpt\latest_test.engine'
    input_size = [640,640]
    batch = 1
    export_to_engine(onnx_path,engine_path,input_size,batch,ch=3,input_shapes=None,fp16=True,dynamic=True,workspace=4,device="cuda:0")