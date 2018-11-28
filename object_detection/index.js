import {
  model_boxes_to_corners,
  model_head,
  model_filter_boxes,
  model_ANCHORS,
} from './preprocess.js';

import class_names from './coco_classes.js';

const DEFAULT_INPUT_DIM = 416;

const DEFAULT_MAX_BOXES = 2048; // TODO: Check note later. 
const DEFAULT_FILTER_BOXES_THRESHOLD = 0.01;
const DEFAULT_IOU_THRESHOLD = 0.4;
const DEFAULT_CLASS_PROB_THRESHOLD = 0.4
const DEFAULT_GRAPH_LOCATION = 'model_js/tensorflowjs_model.pb'
const DEFAULT_WEIGHTS_LOCATION = 'model_js/weights_manifest.json'
const DEFAULT_INPUT = document.getElementById("myCanvas")

/**
 * Downloads a tfjs.Model
 * @param {String} url Trained Model URL
 */
export async function downloadModel(model_url = DEFAULT_GRAPH_LOCATION, weights_url = DEFAULT_WEIGHTS_LOCATION) {
  return await tfjs.loadFrozenModel(model_url, weights_url);
}

/**
 * Given an input image and model, outputs bounding boxes of detected
 * objects with class labels and class probabilities.
 * @param {tfjs.Tensor} input Expected shape (1, 416, 416, 3)
 * Tensor representing input image (RGB 416x416)
 * @param {tfjs.Model} model Model to use
 * @param {Object} [options] Override options for customized
 * models or performance
 * @param {Number} [options.classProbThreshold=0.4] Filter out classes
 * below a certain threshold
 * @param {Number} [options.iouThreshold=0.4] Filter out boxes that
 * have an IoU greater than this threadhold (refer to tfjs.image.nonMaxSuppression)
 * @param {Number} [options.filterBoxesThreshold=0.01] Threshold to
 * filter out box confidence * class confidence
 * @param {Number} [options.maxBoxes=2048] Number of max boxes to
 * return, refer to tfjs.image.nonMaxSuppression. Note: The model
 * itself can only return so many boxes.
 * @param {tfjs.Tensor} [options.modelAnchors=See src/preprocess.js]
 * (Advanced) Model Anchor
 * Boxes, only needed if retraining on a new dataset
 * @param {Number} [options.width=416] (Advanced) If your model's input width is not 416, only if you're using a custom model
 * @param {Number} [options.height=416] (Advanced) If your model's input height is not 416, only if you're using a custom model
 * @param {Number} [options.numClasses=80] (Advanced) If your model has a different number of classes, only if you're using a custom model
 * @param {Array<String>} [options.classNames=See src/coco_classes.js] (Advanced) If your model has non-MSCOCO class names, only if you're using a custom model
 * @returns {Array<Object>} An array of found objects with `className`,
 * `classProb`, `bottom`, `top`, `left`, `right`. Positions are in pixel
 * values. `classProb` ranges from 0 to 1. `className` is derived from
 * `options.classNames`.
 */
async function my_model(
  input = DEFAULT_INPUT,
  model,
  {
    classProbThreshold = DEFAULT_CLASS_PROB_THRESHOLD,
    iouThreshold = DEFAULT_IOU_THRESHOLD,
    filterBoxesThreshold = DEFAULT_FILTER_BOXES_THRESHOLD,
    modelAnchors = model_ANCHORS,
    maxBoxes = DEFAULT_MAX_BOXES,
    width: widthPx = DEFAULT_INPUT_DIM,
    height: heightPx = DEFAULT_INPUT_DIM,
    numClasses = 80,
    classNames = class_names,
  } = {},
) {
  const outs = tfjs.tidy(() => { // Keep as one var to dispose easier
    const activation = model.predict(input);

    const [box_xy, box_wh, box_confidence, box_class_probs ] =
      model_head(activation, modelAnchors, numClasses);

    const all_boxes = model_boxes_to_corners(box_xy, box_wh);

    let [boxes, scores, classes] = model_filter_boxes(
      all_boxes, box_confidence, box_class_probs, filterBoxesThreshold);

    // If all boxes have been filtered out
    if (boxes == null) {
      return null;
    }

    const width = tfjs.scalar(widthPx);
    const height = tfjs.scalar(heightPx);

    const image_dims = tfjs.stack([height, width, height, width]).reshape([1,4]);

    boxes = tfjs.mul(boxes, image_dims);

    return [boxes, scores, classes];
  });

  if (outs === null) {
    return [];
  }

  const [boxes, scores, classes] = outs;

  const indices = await tfjs.image.nonMaxSuppressionAsync(boxes, scores, maxBoxes, iouThreshold)

  // Pick out data that wasn't filtered out by NMS and put them into
  // CPU land to pass back to consumer
  const classes_indx_arr = await classes.gather(indices).data();
  const keep_scores = await scores.gather(indices).data();
  const boxes_arr = await boxes.gather(indices).data();

  tfjs.dispose(outs);
  indices.dispose();

  const results = [];

  classes_indx_arr.forEach((class_indx, i) => {
    const classProb = keep_scores[i];
    if (classProb < classProbThreshold) {
      return;
    }

    const className = classNames[class_indx];
    let [top, left, bottom, right] = [
      boxes_arr[4 * i],
      boxes_arr[4 * i + 1],
      boxes_arr[4 * i + 2],
      boxes_arr[4 * i + 3],
    ];

    top = Math.max(0, top);
    left = Math.max(0, left);
    bottom = Math.min(heightPx, bottom);
    right = Math.min(widthPx, right);

    const resultObj = {
      className,
      classProb,
      bottom,
      top,
      left,
      right,
    };

    results.push(resultObj);
  });

  return results;
}

export default my_model;

