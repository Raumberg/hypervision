// NOT READY

__global__ void nmsKerr(
    float* predictions,
    float conf_thresh,
    int num_preds,
    int num_classes,
    float* filtered_boxes,
    int* box_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_preds) return;

    float max_conf = -1.0f;
    int cls_id = -1;
    
    // Find max confidence class
    for (int i = 4; i < 4 + num_classes; i++) {
        if (predictions[idx * 6 + i] > max_conf) {
            max_conf = predictions[idx * 6 + i];
            cls_id = i - 4;
        }
    }

    if (max_conf < conf_thresh) return;

    // Atomic increment counter
    int count = atomicAdd(box_count, 1);
    
    // Store box data
    filtered_boxes[count * 6 + 0] = predictions[idx * 6 + 0]; // x
    filtered_boxes[count * 6 + 1] = predictions[idx * 6 + 1]; // y
    filtered_boxes[count * 6 + 2] = predictions[idx * 6 + 2]; // w
    filtered_boxes[count * 6 + 3] = predictions[idx * 6 + 3]; // h
    filtered_boxes[count * 6 + 4] = max_conf;
    filtered_boxes[count * 6 + 5] = cls_id;
}