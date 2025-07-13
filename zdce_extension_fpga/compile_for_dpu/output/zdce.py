import sys
import cv2
import numpy as np
import xir
import vart
import os
import time
import threading

def preprocess_image(image_path, fix_scale=128, width=512, height=512):
    """Preprocess image for ZeroDCE++ quantized model on DPU"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    # Resize and convert to RGB
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1] float32
    img = img.astype(np.float32) / 255.0

    # Multiply by fix_scale (usually 128.0 for [-128, 127] range)
    img = img * fix_scale

    # Clip to int8 valid range
    img = np.clip(img, -128, 127).astype(np.int8)

    # Convert to NCHW layout for DPU (1, 3, H, W)
    #img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img

def preprocess_frame(frame, fix_scale=128, width=512, height=512):
    """Preprocess video frame for ZeroDCE++ quantized model on DPU"""
    # Resize and convert to RGB
    img = cv2.resize(frame, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1] float32
    img = img.astype(np.float32) / 255.0

    # Multiply by fix_scale (usually 128.0 for [-128, 127] range)
    img = img * fix_scale

    # Clip to int8 valid range
    img = np.clip(img, -128, 127).astype(np.int8)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # Shape: (1, H, W, 3)

    return img

def process_video(runner, input_video_path, output_video_path, num_threads):
    """Process video frame by frame with the DPU runner"""
    # Use a more reliable video capture method for embedded systems
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.open(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video with FFmpeg")

            
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer (using 512x512 resolution)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 30.0, (512, 512))
    
    # Get model parameters
    in_tensor = runner.get_input_tensors()[0]
    out_tensor = runner.get_output_tensors()[0]
    in_shape = tuple(in_tensor.dims)
    out_shape = tuple(out_tensor.dims)
    fix_out = out_tensor.get_attr("fix_point")
    
    # Create input and output buffers
    input_buffer = np.empty(in_shape, dtype=np.int8)
    output_buffer = np.empty(out_shape, dtype=np.int8)
    
    frame_idx = 0
    total_latency = 0
    last_log_time = time.time()
    
    print(f"Processing video: {input_video_path}")
    print(f"Total frames: {frame_count if frame_count > 0 else 'unknown'}")
    print(f"Original FPS: {fps:.2f}")
    print("Starting processing...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        input_buffer[0] = processed_frame[0]  # Copy to buffer
        
        # Run inference and measure latency
        start_time = time.time()
        job_id = runner.execute_async([input_buffer], [output_buffer])
        runner.wait(job_id)
        latency = (time.time() - start_time) * 1000  # ms
        
        total_latency += latency
        frame_idx += 1
        
        # Print progress every 2 seconds
        current_time = time.time()
        if current_time - last_log_time > 2.0:
            avg_latency = total_latency / frame_idx
            print(f"Processed {frame_idx} frames | Avg latency: {avg_latency:.2f} ms")
            last_log_time = current_time
        
        # Postprocess output
        output_scale = 1 / (2 ** fix_out)
        output_float = output_buffer.astype(np.float32) * output_scale
        output_img = output_float[0]  # Shape (512, 512, 3)
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # Write processed frame to output video
        out.write(output_img)
    
    # Release resources
    cap.release()
    out.release()
    
    # Print summary
    if frame_idx > 0:
        avg_latency = total_latency / frame_idx
        print(f"\nVideo Processing Summary:")
        print(f"Total frames processed: {frame_idx}")
        print(f"Average latency per frame: {avg_latency:.2f} ms")
        print(f"Output video saved to: {output_video_path}")
    else:
        print("Error: No frames processed")

def worker_thread(runner, input_data, output_data, iterations, thread_id, results):
    """Worker thread function for running inferences"""
    # Warm-up run
    job_id = runner.execute_async([input_data], [output_data])
    runner.wait(job_id)
    
    # Performance measurement
    start_time = time.time()
    
    for i in range(iterations):
        job_id = runner.execute_async([input_data], [output_data])
        runner.wait(job_id)
    
    end_time = time.time()
    
    # Store results
    results[thread_id] = {
        'total_time': end_time - start_time,
        'latency': (end_time - start_time) / iterations * 1000,  # ms
        'fps': iterations / (end_time - start_time)
    }
    
    # Run one more inference for output (only from first thread)
    if thread_id == 0:
        job_id = runner.execute_async([input_data], [output_data])
        runner.wait(job_id)
        results['output_data'] = output_data.copy()

def main_image(xmodel_path, input_path, output_path, iterations=100, num_threads=1):
    """Process single image with performance measurements"""
    # Create DPU runners
    graph = xir.Graph.deserialize(xmodel_path)
    root = graph.get_root_subgraph()
    dpus = [c for c in root.toposort_child_subgraph()
            if c.has_attr("device") and c.get_attr("device") == "DPU"]
    
    if not dpus:
        print("No DPU subgraph found!")
        return
    
    runners = []
    for _ in range(num_threads):
        runners.append(vart.Runner.create_runner(dpus[0], "run"))
    
    # Get model parameters
    in_tensor = runners[0].get_input_tensors()[0]
    out_tensor = runners[0].get_output_tensors()[0]
    in_shape = tuple(in_tensor.dims)
    out_shape = tuple(out_tensor.dims)
    fix_in = in_tensor.get_attr("fix_point")
    fix_out = out_tensor.get_attr("fix_point")
    
    print(f"Input shape: {in_shape}, Fix point: {fix_in}")
    print(f"Output shape: {out_shape}, Fix point: {fix_out}")
    print(f"Running with {num_threads} threads, {iterations} iterations per thread")
    
    # Preprocess image
    input_scale = 128
    processed_img = preprocess_image(input_path, input_scale)
    if processed_img is None:
        return
    
    # Create input/output buffers for each thread
    input_buffers = []
    output_buffers = []
    
    for i in range(num_threads):
        input_data = np.zeros(in_shape, dtype=np.int8)
        output_data = np.zeros(out_shape, dtype=np.int8)
        input_data[0] = processed_img[0]
        input_buffers.append(input_data)
        output_buffers.append(output_data)
    
    # Create worker threads
    threads = []
    results = {}
    
    start_time = time.time()
    
    for i in range(num_threads):
        t = threading.Thread(
            target=worker_thread,
            args=(runners[i], input_buffers[i], output_buffers[i], iterations, i, results)
        )
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    del runners
    
    end_time = time.time()
    
    # Calculate performance metrics
    total_inferences = iterations * num_threads
    total_time = end_time - start_time
    
    # Aggregate thread results
    total_fps = 0
    total_latency = 0
    
    print("\nThread Performance:")
    for i in range(num_threads):
        thread_result = results[i]
        print(f"Thread {i}: {thread_result['fps']:.2f} FPS, "
              f"{thread_result['latency']:.2f} ms")
        total_fps += thread_result['fps']
        total_latency += thread_result['latency']
    
    # Overall metrics
    overall_fps = total_inferences / total_time
    avg_latency = total_latency / num_threads
    
    print("\nOverall Performance:")
    print(f"Total inferences: {total_inferences}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Throughput: {overall_fps:.2f} FPS")
    
    # Save output from first thread
    if 'output_data' in results:
        output_scale = 1 / (2 ** fix_out)
        output_float = results['output_data'].astype(np.float32) * output_scale
        output_img = output_float[0]  # Shape (512, 512, 3)
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_img)
        print(f"\nEnhanced image saved to {output_path}")
    else:
        print("Warning: No output image was generated")

def main_video(xmodel_path, input_path, output_path, num_threads=1):
    """Process video file frame by frame"""
    # Create DPU runner
    graph = xir.Graph.deserialize(xmodel_path)
    root = graph.get_root_subgraph()
    dpus = [c for c in root.toposort_child_subgraph()
            if c.has_attr("device") and c.get_attr("device") == "DPU"]
    
    if not dpus:
        print("No DPU subgraph found!")
        return
    
    runner = vart.Runner.create_runner(dpus[0], "run")
    
    # Process video
    process_video(runner, input_path, output_path, num_threads)
    
    # Clean up
    del runner

if __name__ == "__main__":
    # Define video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    
    if len(sys.argv) < 4:
        print("Usage: python3 zero_dce.py <xmodel> <input> <output> [iterations] [threads]")
        print("For images: python3 zero_dce.py zdce.xmodel input.jpg output.jpg 100 2")
        print("For videos: python3 zero_dce.py zdce.xmodel input.mp4 output.mp4 [threads]")
        sys.exit(1)
    
    # Extract arguments
    xmodel = sys.argv[1]
    inp = sys.argv[2]
    outp = sys.argv[3]
    
    # Determine if input is video or image
    is_video = any(inp.lower().endswith(ext) for ext in video_extensions)
    
    if is_video:
        # Video processing mode
        num_threads = 1
        if len(sys.argv) >= 5:
            try:
                num_threads = int(sys.argv[4])
            except ValueError:
                print("Invalid thread count. Using default 1.")
                
        print(f"Processing video: {inp}")
        print(f"Using {num_threads} thread(s)")
        main_video(xmodel, inp, outp, num_threads)
    else:
        # Image processing mode
        iterations = 100
        num_threads = 1
        
        if len(sys.argv) >= 5:
            try:
                iterations = int(sys.argv[4])
            except ValueError:
                print("Invalid iteration count. Using default 100.")
        
        if len(sys.argv) >= 6:
            try:
                num_threads = int(sys.argv[5])
            except ValueError:
                print("Invalid thread count. Using default 1.")
                
        print(f"Processing image: {inp}")
        main_image(xmodel, inp, outp, iterations, num_threads)