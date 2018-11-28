import * as tfjs from '@tensorflow/tfjs';

export const model_ANCHORS = tfjs.tensor2d([
  [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434],
  [7.88282, 3.52778], [9.77052, 9.16828],
]);

export function model_filter_boxes(
  boxes,
  box_confidence,
  box_class_probs,
  threshold
) {
  const box_scores = tfjs.mul(box_confidence, box_class_probs);
  const box_classes = tfjs.argMax(box_scores, -1);
  const box_class_scores = tfjs.max(box_scores, -1);

  const prediction_mask = tfjs.greaterEqual(box_class_scores, tfjs.scalar(threshold)).as1D();

  const N = prediction_mask.size
  // linspace start/stop is inclusive.
  const all_indices = tfjs.linspace(0, N - 1, N).toInt();
  const neg_indices = tfjs.zeros([N], 'int32');
  const indices = tfjs.where(prediction_mask, all_indices, neg_indices);

  return [
    tfjs.gather(boxes.reshape([N, 4]), indices),
    tfjs.gather(box_class_scores.flatten(), indices),
    tfjs.gather(box_classes.flatten(), indices),
  ];
}

/**
 * Given XY and WH tensor outputs of _head, returns corner coordinates.
 * @param {tfjs.Tensor} box_xy Bounding box center XY coordinate Tensor
 * @param {tfjs.Tensor} box_wh Bounding box WH Tensor
 * @returns {tfjs.Tensor} Bounding box corner Tensor
 */
export function model_boxes_to_corners(box_xy, box_wh) {
  const two = tfjs.tensor1d([2.0]);
  const box_mins = tfjs.sub(box_xy, tfjs.div(box_wh, two));
  const box_maxes = tfjs.add(box_xy, tfjs.div(box_wh, two));

  const dim_0 = box_mins.shape[0];
  const dim_1 = box_mins.shape[1];
  const dim_2 = box_mins.shape[2];
  const size = [dim_0, dim_1, dim_2, 1];

  return tfjs.concat([
    box_mins.slice([0, 0, 0, 1], size),
    box_mins.slice([0, 0, 0, 0], size),
    box_maxes.slice([0, 0, 0, 1], size),
    box_maxes.slice([0, 0, 0, 0], size),
  ], 3);
}

// Convert model output to bounding box + prob tensors
export function model_head(feats, anchors, num_classes) {
  const num_anchors = anchors.shape[0];

  const anchors_tensor = tfjs.reshape(anchors, [1, 1, num_anchors, 2]);

  let conv_dims = feats.shape.slice(1, 3);

  // For later use
  const conv_dims_0 = conv_dims[0];
  const conv_dims_1 = conv_dims[1];

  let conv_height_index = tfjs.range(0, conv_dims[0]);
  let conv_width_index = tfjs.range(0, conv_dims[1]);
  conv_height_index = tfjs.tile(conv_height_index, [conv_dims[1]])

  conv_width_index = tfjs.tile(tfjs.expandDims(conv_width_index, 0), [conv_dims[0], 1]);
  conv_width_index = tfjs.transpose(conv_width_index).flatten();

  let conv_index = tfjs.transpose(tfjs.stack([conv_height_index, conv_width_index]));
  conv_index = tfjs.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])
  conv_index = tfjs.cast(conv_index, feats.dtype);

  feats = tfjs.reshape(feats, [conv_dims[0], conv_dims[1], num_anchors, num_classes + 5]);
  conv_dims = tfjs.cast(tfjs.reshape(tfjs.tensor1d(conv_dims), [1,1,1,2]), feats.dtype);

  let box_xy = tfjs.sigmoid(feats.slice([0,0,0,0], [conv_dims_0, conv_dims_1, num_anchors, 2]))
  let box_wh = tfjs.exp(feats.slice([0,0,0, 2], [conv_dims_0, conv_dims_1, num_anchors, 2]))
  const box_confidence = tfjs.sigmoid(feats.slice([0,0,0, 4], [conv_dims_0, conv_dims_1, num_anchors, 1]))
  const box_class_probs = tfjs.softmax(feats.slice([0,0,0, 5],[conv_dims_0, conv_dims_1, num_anchors, num_classes]));

  box_xy = tfjs.div(tfjs.add(box_xy, conv_index), conv_dims);
  box_wh = tfjs.div(tfjs.mul(box_wh, anchors_tensor), conv_dims);

  return [ box_xy, box_wh, box_confidence, box_class_probs ];
}

export function box_intersection(a, b) {
  const w = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  const h = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  if (w < 0 || h < 0) {
    return 0;
  }
  return w * h;
}

export function box_union(a, b) {
  const i = box_intersection(a, b);
  return (a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0]) - i;
}

export function box_iou(a, b) {
  return box_intersection(a, b) / box_union(a, b);
}
