import pyopencl as cl
import numpy as np

# Display available OpenCL platforms and devices
platforms = cl.get_platforms()
for platform in platforms:
    devices = platform.get_devices()
    for device in devices:
        print(f"Platform: {platform.name}, Device: {device.name}")

# Perform a simple vector addition on the GPU to verify OpenCL functionality
try:
    # Select the first platform and device
    platform = platforms[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Define a simple kernel for vector addition
    kernel_code = """
    __kernel void add_vectors(__global const float *a, __global const float *b, __global float *c) {
        int gid = get_global_id(0);
        c[gid] = a[gid] + b[gid];
    }
    """
    program = cl.Program(context, kernel_code).build()

    # Create input and output arrays
    size = 10
    a_np = np.random.rand(size).astype(np.float32)
    b_np = np.random.rand(size).astype(np.float32)
    c_np = np.empty_like(a_np)

    # Create buffers
    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)

    # Execute the kernel
    kernel = program.add_vectors
    kernel.set_args(a_buf, b_buf, c_buf)
    cl.enqueue_nd_range_kernel(queue, kernel, (size,), None)
    queue.finish()

    # Copy result from device to host
    cl.enqueue_copy(queue, c_np, c_buf)
    queue.finish()

    # Display results
    print("Vector A:", a_np)
    print("Vector B:", b_np)
    print("Result (A + B):", c_np)
    print("OpenCL test completed successfully.")

except Exception as e:
    print("OpenCL test failed:", str(e))
