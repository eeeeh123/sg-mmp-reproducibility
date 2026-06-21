"""量化方法模块。"""

from ptq.quant.rtn import quantize_model_rtn, dequantize_tensor_rtn
from ptq.quant.gptq import quantize_model_gptq, apply_gptq_to_model_gpu
from ptq.quant.awq import quantize_model_awq, apply_awq_to_model_gpu
from ptq.quant.smoothquant import compute_smooth_scales, apply_smoothquant_to_model
from ptq.quant.mixed_precision import quantize_model_mixed_precision, apply_mixed_precision_to_model_gpu
