
const tf = require("@tensorflow/tfjs-node");

// Define a simple model for linear regression
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
async function trainModel() {
  console.log("Starting model training...");
  for (let i = 0; i < 500; i++) {
    // Train for 500 epochs
    await model.fit(xs, ys, { epochs: 1 });
    if (i % 100 === 0) {
      console.log(`Epoch ${i}: Loss = ${model.evaluate(xs, ys).dataSync()[0]}`);
    }
  }
  console.log("Model training complete.");
}

// Make a prediction
async function predict() {
  await trainModel();
  const output = model.predict(tf.tensor2d([5], [1, 1]));
  console.log(`Prediction for x = 5: ${output.dataSync()[0]}`);
  // Save the model
  await model.save("file://./my-model");
  console.log("Model saved to ./my-model");
}

predict();
