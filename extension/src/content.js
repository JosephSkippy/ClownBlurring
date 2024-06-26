// This scripts is used to fetch images from the front page of the website and send it into background.js that model will be used to predict the image.


console.log("Content script loaded");
const imageDataMap = {};
/**
 * Loads an image from the specified source URL, renders it to a canvas, and sends its ImageData back.
 * @param {string} src - The source URL of the image to load.
 * @param {Function} sendResponse - The function to send the response back.
 * @returns {Promise<void>} - A promise that resolves when the image is loaded and the response is sent.
 */
const loadImageAndSendDataBack = async (src, sendResponse) => {

    const img = new Image();

    //https://www.youtube.com/watch?v=m6lsF8z0hKk
    //By setting `crossOrigin = "anonymous"` instruct the browser to fetch the image without sending credentials.
    // This allows the image to be used in a canvas without tainting the canvas with a cross-origin exception.

    img.crossOrigin = "anonymous";
    img.onerror = function (e) {
        console.warn(`Could not load image from external source ${src}.`);
        sendResponse({ rawImageData: undefined });
        return;
      };
    
      img.onload = async function (e) {
        // When the image is loaded, render it to a canvas and send its ImageData back
        const canvas = new OffscreenCanvas(img.width, img.height);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, img.width, img.height);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);

        imageDataMap[src] = {
          rawImageData: Array.from(imageData.data),
          width: img.width,
          height: img.height,
        };
    
        sendResponse({
          rawImageData: Array.from(imageData.data),
          width: img.width,
          height: img.height,
        });
      };
    
      img.src = src;
    };

/**
 * Checks the result by loading an image and sending the data back to the runtime.
 * @param {string} msg - The message to be sent.
 * @param {number} index - The index value.
 * @param {string} url - The URL of the image to be loaded.
 */
let checkResult = function (msg, index, url) {
  loadImageAndSendDataBack(url, function (data) {
    chrome.runtime.sendMessage({
      msg: msg,
      index: index,
      url: url,
      data: data,
    });
  });
};


/**
 * Detects and processes images on the page.
 */
let detectImg = function () {
  let images = document.getElementsByTagName("img");
  for (let i = 0; i < images.length; i++) {
    if (images[i].classList.contains("verified")) {
      continue;
    } else {
      // add "verified" class to all images that was verified by the model.
      images[i].classList.add("verified");
      let url = images[i].src;
      // add a message image to be sent to the background script
      checkResult("image", i, url);
    }
  }
};

detectImg();

chrome.runtime.onMessage.addListener(async (request, sender, senderResponse) => {
  console.log(request)
  if (request.detected === 1) {
    console.log("Clown Detected");

    // Get the image data
    const { rawImageData, width, height } = imageDataMap[request.url]// get the url image where for the detected image 
    const imageData = new ImageData(new Uint8ClampedArray(rawImageData), width, height);

    console.log("ImageData Built")

    // Create a canvas to draw the image and apply the mask
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
    
    console.log("Mask shape inside of content.js")
    console.log(typeof request.masks)
    console.log(request.masks)

    // Apply the full masks
    request.masks.forEach((mask2D, maskIndex) => {
      // Convert mask to 8-bit format and resize to match image size
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = width;
      maskCanvas.height = height;
      const maskCtx = maskCanvas.getContext('2d');
      const maskImageData = maskCtx.createImageData(width, height);

      console.log(`Processing full mask ${maskIndex}`);

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const index = y * width + x;
          const value = mask2D[y][x] * 255; // Assuming mask values are 0 or 1
          maskImageData.data[index * 4] = value; // R
          maskImageData.data[index * 4 + 1] = 0; // G
          maskImageData.data[index * 4 + 2] = 0; // B
          maskImageData.data[index * 4 + 3] = value > 0 ? 255 : 0; // A (fully opaque if value is 1)
        }
      }

      maskCtx.putImageData(maskImageData, 0, 0);

      // Draw the mask on the original image
      ctx.globalCompositeOperation = 'source-over';
      ctx.drawImage(maskCanvas, 0, 0);

      console.log("done drawing");
    });

    

    // Replace the original image with the masked image
    const maskedImage = canvas.toDataURL();
    document.querySelector('img').src = maskedImage; // Assuming you want to replace the first image on the page
    console.log("sendback")

    // // Convert mask values to JSON
    // const masksJson = JSON.stringify(request.masks);

    // // Create a download link and trigger the download for the mask values
    // const downloadJsonLink = document.createElement('a');
    // downloadJsonLink.href = 'data:text/json;charset=utf-8,' + encodeURIComponent(masksJson);
    // downloadJsonLink.download = 'masks.json';
    // document.body.appendChild(downloadJsonLink);
    // downloadJsonLink.click();
    // document.body.removeChild(downloadJsonLink);

    // Create a download link and trigger the download for the masked image
    const downloadImageLink = document.createElement('a');
    downloadImageLink.href = maskedImage;
    downloadImageLink.download = 'masked_image.png';
    document.body.appendChild(downloadImageLink);
    downloadImageLink.click();
    document.body.removeChild(downloadImageLink);

    console.log("sendback");
  }
});

// We run the mutationobserver so that our function detectImg can be called whenever there's ajax call
let observer = new MutationObserver(detectImg);

observer.observe(document.body, {
  childList: true,
  subtree: true,
});




