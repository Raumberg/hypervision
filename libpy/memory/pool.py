import pycuda.driver as cuda

class MemoryPool:
    def __init__(self):
        self.input_buffers = [cuda.mem_alloc(1920*1080*3) for _ in range(3)]
        self.output_buffers = [cuda.mem_alloc(640*640*3*4) for _ in range(3)]
        self.current_buffer = 0

    def get_buffers(self):
        buf = self.current_buffer
        self.current_buffer = (self.current_buffer + 1) % 3
        return self.input_buffers[buf], self.output_buffers[buf]