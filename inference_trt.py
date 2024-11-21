import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化cuda
import numpy as np
import torchvision.transforms as standard_transforms
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
import torch
from PIL import Image
import cv2
import numpy as np
# from app.train_software.anomaly_infer.read_utils import get_json, load_image, gen_images, save_image, \
#     draw_score, post_process, get_transform, cv_imwrite, anomaly_map_to_pred_mask

import time
import os
from statistics import mean
def cv_imread(file_path, flag=-1):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)
    return cv_img
def cv_imwrite(out_path, img):
    suffix = '.' + out_path.split('.')[-1]
    cv2.imencode(suffix, img)[1].tofile(out_path)  # 保存带中文的图片，opencv保存的tiff，默认是lzw压缩
def save_image(save_path: str, image: np.ndarray, mask_outline: np.ndarray):
    """保存图片

    Args:
        save_path (str):    保存路径
        image (np.ndarray): 原图
        mask (np.ndarray):  mask
        mask_outline (np.ndarray): mask边缘
        superimposed_map (np.ndarray): 热力图+原图
        pred_score (float): 预测得分. Defaults to 0.0
    """
    figsize = (2 * 9, 9)
    figure, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(image)
    axes[0].set_title("origin")
    axes[1].imshow(mask_outline)
    axes[1].set_title("outlines")
    # axes[3].imshow(superimposed_map)
    # axes[3].set_title("score: {:.4f}".format(pred_score))
    # plt.show()
    # return
    plt.savefig(save_path)
    plt.close()
class Inference(ABC):
    def __init__(self, meta_path: str, openvino_preprocess: bool = False, efficient_ad: bool = False) -> None:
        """
        Args:
            meta_path (str):            超参数路径
            openvino_preprocess (bool): 是否使用openvino图片预处理,只有openvino模型使用
            efficient_ad (bool): 是否使用efficient_ad模型
        """
        super().__init__()
        # 1.超参数
        # self.meta = get_json(meta_path)
        # 2.openvino图片预处理
        self.openvino_preprocess = openvino_preprocess
        self.efficient_ad = efficient_ad
        # 3.transform
        # self.infer_height = self.meta["infer_size"][0]  # 推理时使用的图片大小
        # self.infer_width = self.meta["infer_size"][1]
        self.infer_height = 640
        self.infer_width = 640

    def warm_up(self):
        """预热模型
        """
        # [h w c], 这是opencv读取图片的shape
        x = np.zeros((1, 1, self.infer_height, self.infer_width), dtype=np.float32)
        self.infer(x)

    def to_tensor(self,image):
        """
        Convert a NumPy array image of shape (H, W, C) and range [0, 255]
        to a tensor of shape (C, H, W) and range [0.0, 1.0].
        """
        # Check if the input is a NumPy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a NumPy array.")

        # Check if the image has 3 dimensions (H, W, C)
        if len(image.shape) != 3:
            raise ValueError("Input image must have 3 dimensions (H, W, C).")

        # Convert from (H, W, C) to (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        return image

    def normalize(self,tensor, mean, std):
        """
        Normalize a tensor image with mean and standard deviation.
        Args:
            tensor (numpy.ndarray): Tensor image of shape (C, H, W) to be normalized.
            mean (list): Mean values for each channel.
            std (list): Standard deviation values for each channel.
        """
        # Check if input is a NumPy array
        if not isinstance(tensor, np.ndarray):
            raise ValueError("Input must be a NumPy array.")

        # Check if the tensor has 3 dimensions (C, H, W)
        if len(tensor.shape) != 3:
            raise ValueError("Input tensor must have 3 dimensions (C, H, W).")

        # Check if mean and std are lists of the same length as the number of channels
        if len(mean) != tensor.shape[0] or len(std) != tensor.shape[0]:
            raise ValueError("Mean and std must have the same length as the number of channels.")

        # Normalize each channel
        for channel in range(tensor.shape[0]):
            tensor[channel] = (tensor[channel] - mean[channel]) / std[channel]

        return tensor

    def single(self, image_path: str, save_path: str) -> None:
        """预测单张图片

        Args:
            image_path (str):   图片路径
            save_path (str):    保存图片路径
        """
        save_path = os.path.join(save_path, os.path.splitext(os.path.basename(image_path))[0] + '.jpg')
        # 1.打开图片
        # img_raw = Image.open(image_path).convert('RGB')
        # 2.保存原图高宽
        # img_raw = img_raw.resize((640, 640), Image.LANCZOS)
        # 3.图片预处理
        # transform = standard_transforms.Compose([
        #     standard_transforms.ToTensor(),
        #     standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        # img = transform(img_raw)

        img_raw = cv2.imread(image_path)

        img_raw = cv2.resize(img_raw, (128, 512))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tensor_image = self.to_tensor(img_raw)
        img = self.normalize(tensor_image, mean, std)


        # 4.推理
        start = time.time()
        pred_logits, pred_points = self.infer(img)  # [900, 900] [1]

        # 推理结果
        outputs_scores = torch.nn.functional.softmax(torch.Tensor(pred_logits), -1)[:, :, 1][0]
        outputs_points = torch.Tensor(pred_points[0])
        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        # draw the predictions
        size = 2
        # img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        img_to_draw = img_raw.copy()
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(save_path.replace('.jpg', 'pred{}.jpg'.format(predict_cnt))),
                    img_to_draw)

        end = time.time()
        print("infer time:", (end - start) * 1000, "ms")
        infer_time = end - start

        # 7.保存图片

        return str(infer_time), str(infer_time)

    def multi(self, image_dir: str, save_dir: str = None) -> None:
        """预测多张图片

        Args:
            image_dir (str):    图片文件夹
            save_dir (str, optional): 保存图片路径,没有就不保存. Defaults to None.
        """
        # 0.检查保存路径
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"mkdir {save_dir}")
        else:
            print("保存路径为None,不会保存图片")

        # 1.获取文件夹中图片
        imgs = os.listdir(image_dir)
        imgs = [img for img in imgs if img.endswith(("jpg", "jpeg", "png", "bmp"))]

        # infer_times: list[float] = []
        # scores: list[float] = []
        infer_times: list = []
        scores: list = []
        # 批量推理
        for img in imgs:
            # 2.拼接图片路径
            image_path = os.path.join(image_dir, img)

            # 3.打开图片
            image = load_image(image_path)
            # 保存原图高宽
            # self.meta["image_size"] = [image.shape[0], image.shape[1]]

            # 4.图片预处理
            x = self.transform(image=image)['image']  # [c, h, w]
            # x = np.expand_dims(x, axis=0)               # [c, h, w] -> [b, c, h, w]
            x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
            x = x.astype(dtype=np.float32)

            # 5.推理
            start = time.time()
            anomaly_map, pred_score = self.infer(x)  # [900, 900] [1]

            # 6.后处理,归一化热力图和概率,缩放到原图尺寸 [900, 900] [1]
            # anomaly_map, pred_score = post_process(anomaly_map, pred_score, self.meta)
            #
            # # 7.生成mask,mask边缘,热力图叠加原图
            # # mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
            # mask, mask_outline = gen_images(image, anomaly_map)
            end = time.time()

            infer_times.append(end - start)
            scores.append(pred_score)
            print("pred_score:", pred_score)  # 0.8885370492935181
            print("infer time:", (end - start) * 1000, "ms")

            if save_dir is not None:
                # 7.保存图片
                save_path = os.path.join(save_dir, img)
                # save_image(save_path, image, mask, mask_outline, superimposed_map, pred_score)
                # save_image(save_path, image, mask_outline)

        print("avg infer time: ", mean(infer_times) * 1000, "ms")
        # draw_score(scores, save_dir)
        l = 0
        for i in infer_times:
            l = l + i
        return str(l), ""

# refer https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientnet/infer.py
class TrtInference(Inference):
    def __init__(self, model_path: str, *args, **kwargs) -> None:
        """
        Args:
            model_path (str):   model_path
        """
        super().__init__(*args, **kwargs)
        # 1.载入模型
        self.get_model(model_path)
        # 2.预热模型
        # self.warm_up()

    def destroy(self):
        self.ctx.pop()
        del self.inputs
        del self.outputs
        del self.stream
        self.ctx.detach()  # 2. 实例释放时需要detech cuda上下文
        # 释放所有分配的cuda显存
        for allocation in self.allocations:
            allocation.free()

    def get_model(self, engine_path: str):
        """获取tensorrt模型

        Args:
            engine_path (str):  模型路径

        """
        cuda.init()
        device = cuda.Device(0)
        self.ctx = device.make_context()
        # async stream
        self.stream = cuda.Stream()
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.INFO)
        self.logger.min_severity = trt.Logger.Severity.VERBOSE

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []  # inputs
        self.outputs = []  # outputs
        self.allocations = []  # inputs&outputs cuda memorys
        for i in range(self.engine.num_bindings):
            is_input = False

            if trt.__version__ < "8.5":
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.context.get_binding_shape(i)
                if shape[0] < 0 and is_input:
                    assert self.engine.num_optimization_profiles > 0
                    profile_shape = self.engine.get_profile_shape(0, name)
                    assert len(profile_shape) == 3  # min,opt,max
                    # Set the *min* profile as binding shape
                    self.context.set_binding_shape(i, profile_shape[0])
                    shape = self.context.get_binding_shape(i)
            else:
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    is_input = True
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.context.get_tensor_shape(name)
                if shape[0] < 0 and is_input:
                    assert self.engine.num_optimization_profiles > 0
                    profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                    assert len(profile_shape) == 3  # min,opt,max
                    # Set the *min* profile as tensor shape
                    self.context.set_input_shape(name, profile_shape[0])
                    shape = self.context.get_tensor_shape(name)

            if is_input:
                self.batch_size = shape[0]

            dtype = np.dtype(trt.nptype(dtype))
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)  # allocate cuda memory
            host_allocation = None if is_input else np.zeros(shape, dtype)  # allocate memory
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)  # allocate cuda memory
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

            print("{} '{}' with shape {} and dtype {}".format(
                "Input:" if is_input else "Output:",
                binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self, i: int = 0):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :params:
            i: the index of input

        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[i]["shape"], self.inputs[i]["dtype"]

    def output_spec(self, i: int = 0):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :params:
            i: the index of input

        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[i]["shape"], self.outputs[i]["dtype"]

    def infer(self, image: np.ndarray) -> tuple:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            tuple[np.ndarray]: anomaly_map, score
        """

        # 1.推理
        # Process I/O and execute the network
        # 将内存中的图片移动到显存上                             将图片内存变得连续
        cuda.memcpy_htod_async(self.inputs[0]['allocation'], np.ascontiguousarray(image), self.stream)
        # cpu memory to gpu memory
        self.ctx.push()
        self.context.execute_async_v2(bindings=self.allocations, stream_handle=self.stream.handle)  # infer
        self.ctx.pop()
        cuda.Context.pop()

        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(self.outputs[i]["host_allocation"], self.outputs[i]["allocation"],
                                   self.stream)  # gpu memory to cpu memory
        # syncronize threads
        self.stream.synchronize()
        # patchcore有两个输出结果 [1]代表anomaly_map [0]代表pred_score
        pred_points = self.outputs[1]["host_allocation"]
        pred_logits = self.outputs[0]["host_allocation"]
        return pred_logits, pred_points

if __name__ == "__main__":
    model_path = r"D:\code\CrowdCounting-P2PNet\ckpt\latest_test.engine"
    meta_path = r"\\192.168.205.91\g\AI小组\数据集\卫品图片\倍舒特床垫\原始图\正样本\unsupervised_work_dirs_09.15-17.36\weights/metadata.json"
    image_path = r"D:\0118_134329_891.bmp"
    # image_dir = r"\\192.168.205.91\g\AI小组\数据集\卫品图片\倍舒特床垫\原始图\CCD1"
    save_path = r"D:\code\CrowdCounting-P2PNet\result"
    # save_dir = r"\\192.168.205.91\g\AI小组\数据集\卫品图片\倍舒特床垫\原始图\正样本\unsupervised_work_dirs_09.15-17.36/result"
    infer = TrtInference(model_path=model_path, meta_path=meta_path, efficient_ad=False)
    infer.single(image_path=image_path, save_path=save_path)
    # infer.destroy()
    # infer.multi(image_dir=image_dir, save_dir=save_dir)