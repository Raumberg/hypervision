from termcolor import colored
from typing import Tuple, List, Dict
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np

class TensorRTEngine:
    """
    Handles TensorRT engine initialization and inference
    """
    
    def __init__(self, engine_path: str):
        """
        Initialize TensorRT engine
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        print(colored("[OKAY] TensorRT engine loaded", "green"))

    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """
        Load TensorRT engine from file
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        try:
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                engine = runtime.deserialize_cuda_engine(f.read())
                if not engine:
                    raise ValueError("Failed to deserialize engine")

                print(colored("\nEngine Details:", "cyan"))
                for index in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(index)
                    dtype = engine.get_tensor_dtype(name)
                    shape = engine.get_tensor_shape(name)
                    mode = "INPUT" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
                    print(f"{mode} | {name} | {dtype} | {shape}")
                return engine
        except Exception as e:
            print(colored(f"\n[ERROR] Engine loading failed: {str(e)}", "red"))
            raise

    def _allocate_buffers(self) -> Tuple[List, List, List, cuda.Stream]:
        """Allocate memory buffers for TensorRT inference"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if is_input:
                inputs.append({'name': name, 'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'name': name, 'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, input_tensor: np.ndarray) -> List[Dict]:
        """Perform inference with current input tensor"""
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        
        for i in range(len(self.inputs)):
            self.context.set_tensor_address(self.inputs[i]['name'], int(self.inputs[i]['device']))
        for i in range(len(self.outputs)):
            self.context.set_tensor_address(self.outputs[i]['name'], int(self.outputs[i]['device']))

        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs