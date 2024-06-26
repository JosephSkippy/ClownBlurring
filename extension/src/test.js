const tf = require('@tensorflow/tfjs');

// Create a tensor with 10 values, all set to 0.00
const tensor = tf.ones([10]);

// Verify the elements
tensor.data().then(data => {
  console.log("Tensor elements:", data); // Should log an array of 0.00
}).catch(error => {
  console.error("Error retrieving tensor data:", error);
});

// Sum the elements
const sumTensor = tf.sum(tensor);
sumTensor.data().then(sum => {
  console.log("Sum of tensor elements:", sum); // Should be 0.00
}).catch(error => {
  console.error("Error getting sum of tensor data:", error);
});