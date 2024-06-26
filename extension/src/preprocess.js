//https://github.com/Hyuto/yolov8-tfjs/blob/master/src/utils/detect.js
//https://github.com/ultralytics/yolov5/discussions/12834
//https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Segmentation-ONNXRuntime-Python/main.py
import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBoxes";


/**
 * Preprocesses the source image by resizing, padding, and normalizing it.
 * 
 * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|ImageData|tf.Tensor} source - The source image to preprocess.
 * @param {number} modelWidth - The desired width of the model input.
 * @param {number} modelHeight - The desired height of the model input.
 * @returns {Array} - An array containing the preprocessed input tensor, xRatio, and yRatio.
 */
// const preprocess = (source, modelWidth, modelHeight) => {
//     let xRatio, yRatio; // ratios for boxes
  
//     const input = tf.tidy(() => {
//       const img = tf.browser.fromPixels(source);

//       console.log(img.shape)
  
//       // padding image to square => [n, m] to [n, n], n > m
//       const [h, w] = img.shape.slice(0, 2); // get source width and height
//       const maxSize = Math.max(w, h); // get max size
//       const imgPadded = img.pad([
//         [0, maxSize - h], // padding y [bottom only]
//         [0, maxSize - w], // padding x [right only]
//         [0, 0],
//       ]);
  
//       xRatio = maxSize / w; // update xRatio
//       yRatio = maxSize / h; // update yRatio
  
//       return tf.image
//         .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
//         .div(255.0) // normalize
//         .expandDims(0); // add batch
//     });
  
//     return [input, xRatio, yRatio];
//   };

const preprocess = (source, modelWidth, modelHeight) => {
  let ratio, padW, padH; // ratios and padding for boxes

  const input = tf.tidy(() => {
      const img = tf.browser.fromPixels(source);

      // Get original image shape
      const [h, w] = img.shape.slice(0, 2);
   

      // Calculate the resize ratio
      const r = Math.min(modelHeight / h, modelWidth / w);
      ratio = [r, r];

      // Calculate new unpadded shape
      const newUnpad = [Math.round(w * r), Math.round(h * r)];


      // Calculate padding
      padW = (modelWidth - newUnpad[0]) / 2;
      padH = (modelHeight - newUnpad[1]) / 2;

      // Resize the image
      let resizedImg = tf.image.resizeBilinear(img, newUnpad);

      console.log("resizedImg")
      console.log(resizedImg.shape)

      // Pad the image
      const top = Math.floor(padH);
      const bottom = Math.ceil(padH);
      const left = Math.floor(padW);
      const right = Math.ceil(padW);

      // Adjust padding to ensure final dimensions are exactly 256x256
      const paddedImg = tf.pad(resizedImg, [[top, bottom], [left, right], [0, 0]], 114);

      // Ensure the final dimensions are exactly 256x256
      const finalImg = tf.image.resizeBilinear(paddedImg, [modelHeight, modelWidth]);

      // Normalize the image
      const normalizedImg = finalImg.div(255.0);

      // Add batch dimension
      const imgProcess = normalizedImg.expandDims(0);

      console.log("imgProcess")
      console.log(imgProcess.shape)

      return imgProcess;
  });

  console.log("ratio")
  console.log(ratio)

  console.log("padW")
  console.log(padW)

  console.log("ratio")
  console.log(padH)

  return [input, ratio, [padW, padH]];
};



async function postprocessing(preds, im0, ratio, pad_w, pad_h, conf_threshold=0.25, iou_threshold=0.45, nm=32){

  im0 = tf.browser.fromPixels(im0);
  const im0_shape = im0.shape.slice(0, 2);

  // x (1, 37, some number)
  //  x = (batch_size, num_classes + 4 + num_masks, num_boxes)
  let x = preds[0]; // get output1
  let protos = preds[1]; // get output2

  console.log("x original shape", x.shape)
  console.log("protos original shape", protos.shape)

  // Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
  x = x.transpose([0, 2, 1]);

  console.log("x transposed orignal shape", x.shape)

  let shape = x.shape;
  // Calculate the end index
  let end = shape[shape.length - 1] - nm; // Equivalent to -nm in Python

  // Extract the confidence scores (assuming they are in the 5th column)
  const confScores = x.slice([0, 0, 4], [-1, -1, end - 4]);

  console.log("confScores")
  console.log(confScores.shape)
  // Find the maximum confidence score for each anchor
  const maxConfScores = confScores.max(-1);
  
  console.log("MaxconfScores")
  console.log(maxConfScores.shape)

  // Create a mask for filtering based on the confidence threshold
  const mask = maxConfScores.greater(conf_threshold);


  console.log("mask")
  console.log(mask.shape)

  // Apply the mask to filter the predictions
  x = await tf.booleanMaskAsync(x, mask);
  console.log(typeof x)
  console.log("x after boolean mask")
  console.log(x.shape)

  shape = x.shape;
  // Calculate the end index
  end = shape[shape.length - 1] - nm; // Equivalent to -nm in Python

  // Extract the components
  let boxes = x.slice([0, 0], [-1, 4]);  // x[..., :4]
  let scores = x.slice([0, 4], [-1, end - 4]).max(-1, true);  // np.amax(x[..., 4:-nm], axis=-1)
  const classes = x.slice([0, 4], [-1, end - 4]).argMax(-1, true).expandDims(-1);  // np.argmax(x[..., 4:-nm], axis=-1)

  console.log("boxes")
  console.log(boxes.shape)

  console.log("scores")
  console.log(scores.shape)

  console.log("classes")
  console.log(classes.shape)
  
  let start = [0, shape[1] - nm];
  let size = [-1, nm]; // [-1, 32]
  
  let masks = x.slice(start, size);
  console.log("masked")
  console.log(masks.shape)
  // Concatenate the results


  x = tf.concat([boxes, scores, classes, masks], -1);

  console.log("x after concat")
  console.log(x.shape)

  // Slicing the first 4 columns (x[:, :4])
  boxes = x.slice([0, 0], [-1, 4]);

  // Slicing the 5th column (x[:, 4])
  scores = x.slice([0, 4], [-1, 1]).squeeze(-1);

  console.log("squeenzed scores")
  console.log(scores.shape)


//    Perform Non-Maximum Suppression
  const selectedIndices = await tf.image.nonMaxSuppressionAsync(
  boxes, scores, boxes.shape[0], iou_threshold, conf_threshold);
  console.log("selectedIndices")
  console.log(selectedIndices.shape)

  selectedIndices.data().then(data => {
    console.log("selectedIndices data:", data);
  }).catch(error => {
    console.error("Error converting selectedIndices to data:", error);
  });

  x = tf.gather(x, selectedIndices);

  console.log("x after gather")
  console.log(x.shape)


  
// Up to this point all correct and validation. Althought JS and PT have some difference but is ok. Major milestone!!!

  // Decode and return
  if (x.shape[0] > 0) {
    // Bounding boxes format change: cxcywh -> xyxy
    // Convert cxcywh to xyxy
    x = x.arraySync(); // Convert tensor to array for easier manipulation

    for (let i = 0; i < x.length; i++) {
      x[i][0] -= x[i][2] / 2; // x1 = cx - w/2
      x[i][1] -= x[i][3] / 2; // y1 = cy - h/2
      x[i][2] += x[i][0];     // x2 = x1 + w
      x[i][3] += x[i][1];     // y2 = y1 + h
    }
console.log("This is x array")
console.log((x))

    // x = tf.tensor2d(x); // Convert array back to tensor don't need to convert back to tensor

    // x.data().then(data => {
    //   console.log("x cxcywh -> xyxy:", data);
    // }).catch(error => {
    //   console.error("Error converting selectedIndices to data:", error);
    // });
  
  

    console.log("4 da tian wang")
    console.log(pad_w, pad_h, pad_w, pad_h, ratio)
    // got problem here
  // Assuming x is a 2D array and pad_w, pad_h, and ratio are defined
    let minRatio = Math.min(...ratio);

    console.log(typeof x)
    console.log(x.shape)

    // First step: Subtract pad_w and pad_h
    for (let i = 0; i < x.length; i++) {
      x[i][0] -= pad_w; // x1
      x[i][1] -= pad_h; // y1
      x[i][2] -= pad_w; // x2
      x[i][3] -= pad_h; // y2
    }
    console.log("x array after padding", x)
    

    // Second step: Divide by minRatio
    for (let i = 0; i < x.length; i++) {
      x[i][0] /= minRatio; // x1
      x[i][1] /= minRatio; // y1
      x[i][2] /= minRatio; // x2
      x[i][3] /= minRatio; // y2
    }

    console.log("x array after minratio", x)

    
    // Bounding boxes boundary clamp
    for (let i = 0; i < x.length; i++) {
      x[i][0] = Math.max(0, Math.min(x[i][0], im0_shape[1])); // Clamp x1
      x[i][2] = Math.max(0, Math.min(x[i][2], im0_shape[1])); // Clamp x2
      x[i][1] = Math.max(0, Math.min(x[i][1], im0_shape[0])); // Clamp y1
      x[i][3] = Math.max(0, Math.min(x[i][3], im0_shape[0])); // Clamp y2
    }
      

    console.log("x array after boundary", x)

    let masksIn = x.map(row => row.slice(6));
    
    let bboxes = x.map(row => row.slice(0, 4));
    console.log(bboxes); // [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]]


    masks = await processMask(protos, masksIn, bboxes, im0.shape);

    

    // let segments = masks2segments(masks);


    console.log("masks2segments mask done")


    boxes = x.slice([0, 0], [-1, 6]);

    console.log("returned post processing done")

    return [boxes, masks]  // boxes, segments, masks
  }
  else{
    return [], [], []  // boxes, segments, masks
  }

}


async function processMask(protos, masksIn, bboxes, im0Shape) {

protos = protos.slice([0, 0, 0, 0], [1, -1, -1, -1]);
  // Remove the first dimension to get a 3D tensor
protos = protos.squeeze([0]);

masksIn = tf.tensor(masksIn);
bboxes = tf.tensor(bboxes);


const [mh, mw, c] = protos.shape;

console.log("maskin shape")
console.log(masksIn.shape)

console.log("protos shape")
console.log(protos.shape)

console.log("bboxes shape")
console.log(bboxes.shape)

console.log("im0Shape shape")
console.log(im0Shape)

const reshapedProtos = protos.reshape([c, mh * mw]);

console.log("reshapedProtos")
console.log(reshapedProtos.shape)

// Perform matrix multiplication
let masks = tf.matMul(masksIn, reshapedProtos);

console.log("matmul done")
console.log(masks.shape)

// Reshape the result to [-1, mh, mw]
masks = masks.reshape([-1, mh, mw]);

console.log("reshape done")
console.log(masks.shape)

// Transpose the result to [mh, mw, -1]
masks = masks.transpose([1, 2, 0]);

console.log("transpose done protos")

console.log(masks.shape)
masks.data().then(data => {
  console.log("transpose done protos:", data);
}).catch(error => {
  console.error("Error converting selectedIndices to data:", error);
});





// Rescale the mask to the original image shape
masks = await scaleMask(masks, im0Shape);

console.log("scale mask done")
console.log(masks.shape)

masks.data().then(data => {
  console.log("scale done:", data);
}).catch(error => {
  console.error("Error converting selectedIndices to data:", error);
});



// Transpose the mask
masks = masks.transpose([2, 0, 1]);  // HWN -> NHW

console.log("scale mask done")
console.log(masks.shape)

// Crop the mask to the bounding boxes
masks = await cropMask(masks, bboxes);

console.log("crop mask done")

console.log(typeof masks)
console.log(masks.shape)

// Apply threshold
masks = masks.greater(0.5);

console.log("Mask threshold done")
console.log(masks.shape)

let x = masks.arraySync(); 

console.log("final mask!!")
console.log(x)

return masks;
}


async function scaleMask(masks, im0Shape, ratioPad = null) {
const im1Shape = masks.shape.slice(0, 2);
let gain, pad;

if (ratioPad === null) {
  gain = Math.min(im1Shape[0] / im0Shape[0], im1Shape[1] / im0Shape[1]);  // gain = old / new
  pad = [(im1Shape[1] - im0Shape[1] * gain) / 2, (im1Shape[0] - im0Shape[0] * gain) / 2];  // wh padding
} else {
  pad = ratioPad[1];
}

// Calculate tlbr of mask
const top = Math.round(pad[1] - 0.1);
const left = Math.round(pad[0] - 0.1);
const bottom = Math.round(im1Shape[0] - pad[1] + 0.1);
const right = Math.round(im1Shape[1] - pad[0] + 0.1);

if (masks.shape.length < 2) {
  throw new Error(`"len of masks shape" should be 2 or 3, but got ${masks.shape.length}`);
}

// Crop the mask
masks = masks.slice([top, left, 0], [bottom - top, right - left, -1]);

// Resize the mask
masks = tf.image.resizeBilinear(masks, [im0Shape[0], im0Shape[1]], true);

if (masks.shape.length === 2) {
  masks = masks.expandDims(-1);
}

return masks;
}


// Function to crop masks
async function cropMask(masks, boxes) {
const [n, h, w] = masks.shape;

// Extract bounding box coordinates
const [x1, y1, x2, y2] = tf.split(boxes.expandDims(2), 4, 1);

// Create range tensors
const r = tf.range(0, w, 1, 'int32').expandDims(0).expandDims(0);
const c = tf.range(0, h, 1, 'int32').expandDims(0).expandDims(2);

// Apply the mask
const mask = tf.logicalAnd(
tf.logicalAnd(r.greaterEqual(x1), r.less(x2)),
tf.logicalAnd(c.greaterEqual(y1), c.less(y2))
);

return masks.mul(mask.cast(masks.dtype));
}

function masks2segments(masks) {
  /**
   * It takes a list of masks (n, h, w) and returns a list of segments (n, xy).
   *
   * Args:
   *     masks (Array): the output of the model, which is a tensor of shape (batch_size, 160, 160).
   *
   * Returns:
   *     segments (Array): list of segment masks.
   */
  console.log("masks2segments")
  console.log(masks.shape)
  console.log(typeof masks)

  if (typeof masks !== 'object' || !Array.isArray(masks)) {
    // Convert masks to a standard JavaScript array
    masks = masks.arraySync ? masks.arraySync() : Array.from(masks);
}

console.log("masks2segments array Sync done")
console.log(masks.shape)
console.log(typeof masks)

  let segments = [];
  for (let i = 0; i < masks.length; i++) {
      let mask = masks[i];
      console.log(`Processing mask ${i} with shape: [${mask.length}, ${mask[0].length}]`);
      let contours = findContours(mask);

      console.log("contours found")


      if (contours.length > 0) {
          // Find the largest contour
          let largestContour = contours.reduce((maxContour, contour) => {
              return contour.length > maxContour.length ? contour : maxContour;
          }, contours[0]);
          segments.push(largestContour.map(point => [point[1], point[0]])); // Convert to (x, y) format
      } else {
          segments.push([]); // No segments found
      }
  }
  return segments;
}

function findContours(mask) {
/**
* A simple contour detection algorithm.
*
* Args:
*     mask (Array): a 2D array representing the mask.
*
* Returns:
*     contours (Array): list of contours found in the mask.
*/
let contours = [];
let visited = new Set();

function isInside(x, y) {
  return x >= 0 && y >= 0 && x < mask.length && y < mask[0].length;
}

function dfs(x, y, contour) {
  let stack = [[x, y]];
  while (stack.length > 0) {
      let [cx, cy] = stack.pop();
      if (!isInside(cx, cy) || visited.has(`${cx},${cy}`) || mask[cx][cy] === 0) {
          continue;
      }
      visited.add(`${cx},${cy}`);
      contour.push([cx, cy]);
      stack.push([cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]);
  }
}

for (let x = 0; x < mask.length; x++) {
  for (let y = 0; y < mask[0].length; y++) {
      if (mask[x][y] === 1 && !visited.has(`${x},${y}`)) {
          let contour = [];
          dfs(x, y, contour);
          contours.push(contour);
      }
  }
}

return contours;
}

  export const detect = async (source, model) => {
    const [modelWidth, modelHeight] = [256, 256]; // model input size
  
    tf.engine().startScope(); // start scoping tf engine
    const [input, ratio, [padW, padH]] = preprocess(source, modelWidth, modelHeight); // preprocess image

    const res = model.execute(input); // inference model

    console.log("res")
    console.log(res[0].shape)
    console.log(res[1].shape)

    const [boxes, masks] = await postprocessing(res, source, ratio, padW, padH); // postprocess model output


    const anyNonZero = await masks.any().data();
    const detected = anyNonZero[0] ? 1 : 0;
    console.log("detected")
    console.log(detected)



    return [boxes, masks, detected] 

  };

  
    
    
  
