const fs = require("fs");

// ---- Load data ----
const images = JSON.parse(fs.readFileSync("train_images.txt", "utf8"));
const labels = JSON.parse(fs.readFileSync("train_labels.txt", "utf8"));
let weights = JSON.parse(fs.readFileSync("weights.txt", "utf8"));
let biases = JSON.parse(fs.readFileSync("biases.txt", "utf8"));
const method = "sigmoid"

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function relu(x) {
    return Math.max(0, x);
}

function activation(prevLayer, weight, bias, last) {
    let node = [];
    for (let i = 0; i < bias.length; i++) {
        let sum = 0;
        for (let j = 0; j < prevLayer.length; j++) {
            sum += prevLayer[j] * weight[i][j];
        }
        sum += bias[i];
        if (last === true){
            node.push(sigmoid(sum));
        } else if (last === false){
            node.push(relu(sum));
        }
    }
    return node;
}

function run(image, weights, biases) {
    let NN = image
    for (let i = 0; i < biases.length; i++)
        if (method === "relu"){
            if (i === biases.length-1) {
                NN = activation(NN, weights[i], biases[i], true);
            } else {
                NN = activation(NN, weights[i], biases[i], false);
            }
        } else if (method === "sigmoid"){
            NN = activation(NN, weights[i], biases[i], true);
        }
    return NN; // return final output layer
}

// ---- Evaluation ----
function evaluate(images, labels, weights, biases) {
    const startTime = Date.now();
    let correct = 0;

    for (let i = 0; i < images.length; i++) {
        let output = run(images[i], weights, biases);

        // Predicted label = index of max activation
        let predicted = output.indexOf(Math.max(...output));
        if (predicted === labels[i]) {
            correct++;
        }

        // Progress info
        if ((i + 1) % 10 === 0 || i === images.length - 1) {
            let elapsed = (Date.now() - startTime) / 1000; // seconds
            let avgPerSample = elapsed / (i + 1);
            let remaining = images.length - (i + 1);
            let eta = avgPerSample * remaining;
            let accuracy = (correct / (i + 1)) * 100;

            console.log(
                `Sample ${i + 1}/${images.length} | ` +
                `elapsed: ${elapsed.toFixed(1)}s | ` +
                `ETA: ${eta.toFixed(1)}s | ` +
                `accuracy: ${accuracy.toFixed(2)}%`
            );
        }
    }

    console.log(`Final Accuracy: ${(correct / images.length * 100).toFixed(2)}%`);
}

// ---- Run evaluation ----
evaluate(images, labels, weights, biases);