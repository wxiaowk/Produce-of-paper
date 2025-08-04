from model.CNN_model import DeepJSCC, DeepJSCC1
from model.model_unquan import DeepJSCC as DeepJSCC_unquantized
import torch
from thop import profile
if __name__ == "__main__":
    # def __init__(self, c, channel_type, snr, n_bit):
    model = DeepJSCC1(c=8, channel_type= 'awgn', snr=19, n_bit= 8)
    # model = DeepJSCC_unquantized(c= 8, channel_type= 'awgn', snr=12)
    input = torch.randn(64, 3, 32, 32) #
    Flops, params = profile(model, inputs=(input,)) # macs
    print('Flops: % .4f MB'%(Flops / 1000000))# 计算量
    print('params参数量: % .4f KB'% (params / 1000))   # 参数量：等价与上面的summary输出的Total params值
# def save_model(model, filepath):
#     torch.save(model.state_dict(), filepath)
#
# def get_file_size(filepath):
#     import os
#     size = os.path.getsize(filepath)
#     return size
#
# def quantize_model(model):
#     # 使用 per-tensor 量化配置
#     model.qconfig = torch.quantization.QConfig(
#         activation=torch.quantization.default_observer,
#         weight=torch.quantization.default_weight_observer
#     )
#     torch.quantization.prepare(model, inplace=True)
#     # 可以在这里添加校准数据，使模型量化更准确
#     # 例如，使用几批训练数据通过模型
#     # model(input_data)
#     torch.quantization.convert(model, inplace=True)
#     return model
#
# if __name__ == "__main__":
#     # 原始32位模型
#     model_32 = DeepJSCC(c=40, channel_type='awgn', snr=9, n_bit=32)
#     save_model(model_32, "model_32bit.pth")
#     size_32bit = get_file_size("model_32bit.pth")
#     print('32-bit model size: %.2f KB' % (size_32bit / 1024))
#
#     # 8位量化模型
#     model_8 = DeepJSCC(c=40, channel_type='awgn', snr=9, n_bit=8)
#     model_8 = quantize_model(model_8)
#     save_model(model_8, "model_8bit.pth")
#     size_8bit = get_file_size("model_8bit.pth")
#     print('8-bit model size: %.2f KB' % (size_8bit / 1024))