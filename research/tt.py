from torch.utils.cpp_extension import load

lltm_cpp = load(name='lltm_scpp', sources=['lltm.cpp', 'lltm_cuda_kernel.cu'])