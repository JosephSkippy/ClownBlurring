
//https://github.com/hugozanini/TFJS-object-detection
//https://blog.tensorflow.org/2021/01/custom-object-detection-in-browser.html
//https://github.com/ultralytics/ultralytics/blob/e30b7c24f22b98dcf30ca5d23871659f861acadd/examples/YOLOv8-Segmentation-ONNXRuntime-Python/main.py contains the whole pipeline

import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import { detect } from "./preprocess"; 

const TEN_SECONDS_IN_MS = 10000;

class InstanceSegmentation {
  constructor() {
    this.loadModel();
  }

  async loadModel() {
    console.log("Loading model...");
    try {
      let json = chrome.runtime.getURL("model/model.json");
      chrome.storage.local.set({ model: json });
      this.model = await loadGraphModel(json);
      console.log("Model loaded successfully");
    } catch (e) {
      console.error("Unable to load model", e);
    }
  }

  async CheckImage(index, data) {
    if (!this.model) {
      console.log("Waiting for model to load...");
      setTimeout(() => {
        this.CheckImage(index, data);
      }, TEN_SECONDS_IN_MS);
      return;
    }
    console.log("Predicting...");

    let imageData = new ImageData(
      Uint8ClampedArray.from(data.rawImageData),
      data.width,
      data.height
    );

    // Assuming detect is an async function
    let [boxes, masks, detected]  = await detect(imageData, this.model);


        // convert from tensor to array
        let masksArray;
        if (masks instanceof tf.Tensor) {
          masksArray = await masks.array();
        } else {
          masksArray = masks;
        }
    



    // Do something with processedImages if needed
    return [boxes, masksArray, detected];
  }
}


console.log("Background script running");

const instanceSegmentation = new InstanceSegmentation();

// Listen for messages from the content script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.msg === "image") {
    instanceSegmentation.CheckImage(request.index, request.data).then(([boxes, masksArray, detected]) => {

      // Send the segmentation results back to the content script
      let returnMessage = { index: request.index, url: request.url, masks: masksArray, detected:detected};

      console.log("Sending segmentation results back to content script");
      chrome.tabs.sendMessage(sender.tab.id, returnMessage);

      sendResponse({ status: "success" });
    }).catch(error => {
      console.error("Error processing image:", error);
      sendResponse({ status: "error", error: error.message });
    });

    // Return true to indicate that the response will be sent asynchronously
    return true;
  } else if (request.msg === "toggle_background") {
    if (request.enabled) {
      console.log("Background script enabled");
      // Start any necessary background tasks
    } else {
      console.log("Background script disabled");
      // Stop any necessary background tasks
    }
  }
});


// console.log("Background script running");

// const instanceSegmentation = new InstanceSegmentation();

// // Listen for messages from the content script or popup
// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   if (request.msg === "image") {
//     instanceSegmentation.CheckImage(request.index, request.data).then(([boxes, segments, masks] ) => {

//       console.log("Processed Images", processedImages);
//       // Send the processed images back to the content script
//       chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
//         if (tabs && tabs.length > 0) {
//           chrome.tabs.sendMessage(tabs[0].id, {
//             msg: "processedImages",
//             image: processedImage.src
//           });
//         } else {
//           console.error("No active tabs found");
//         }
//       });
//       sendResponse({ status: "success" });
//     }).catch(error => {
//       console.error("Error processing image:", error);
//       sendResponse({ status: "error", error: error.message });
//     });

//     // Return true to indicate that the response will be sent asynchronously
//     return true;
//   } else if (request.msg === "toggle_background") {
//     if (request.enabled) {
//       console.log("Background script enabled");
//       // Start any necessary background tasks
//     } else {
//       console.log("Background script disabled");
//       // Stop any necessary background tasks
//     }
//   }
// });

// Check the initial state from Chrome storage
chrome.storage.sync.get(['backgroundEnabled'], function(result) {
  if (result.backgroundEnabled) {
    console.log("Background script is initially enabled");
    // Start any necessary background tasks
  } else {
    console.log("Background script is initially disabled");
    // Stop any necessary background tasks
  }
});