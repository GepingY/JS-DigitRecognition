//28x28input 128node 128node 10node for output
//Backpropagation + Sigmoid + squared error + Stochastic Gradient Decent
//z[2] means z value of layer 2; z[1] means z value of layer 1
//∂C/∂Wi = (∂C/∂a)*(∂a/∂z)*(∂z/∂wi)
//This code has kept the functions for Relu + Sigmoid + SquareError instead of pure Sigmoid in commented areas
//const images = [[]] //structure of [[p1, p2, p3, p4], [p1, p2, p3]] every D1 is image arrays, every D2 is pixels of the image
//const labels = [] //normal array, each corrsponding to the and written number represented by the image of the same index in var "images"
const fs = require("fs");
const images = JSON.parse(fs.readFileSync("train_images.txt", "utf8")); //data file
const labels = JSON.parse(fs.readFileSync("train_labels.txt", "utf8")); //data file
const weightsOutput = "weights.txt" //name of the file for weights to save/overide to auto create if not already exist
const biasesOutput = "biases.txt" //name of the file for biases to save/overide to auto create if not already exist

console.log(`Loaded ${images.length} images and ${labels.length} labels`);

const step = 0.01 //rate of step size in learnnig, η 
const startTime = Date.now(); //init time stamp
const method = "sigmoid" //change to relu if want to use Relu for Hidden Layers
const Node = 128
const sizeInitFactor = 0.5 //using extreme value like lower than 0.1 or 10 will result in vanishing gradient which stops the trainning, 0.1 results in slow
const sizeInitTrend = 0.5 //for trend of 0, there won't be any negative value, higher than 0.5 for trending towards positive, vice versa(it's not working when at 0)
const sizeInit = function(Trend, Factor){
    return (Math.random() - Trend) * Factor
}

let weights = []

for (let i = 0; i < 3; i++){ //this for loop is used to manually adjust weights in the beginning, for example making the edge area of the piture have close to 0 weights since we know that they don't matter
    weights.push([])
    for (let ii = 0; ii < Node; ii++){
        weights[i].push([])
        if (i === 0){
            for (let iii = 0; iii < 28*28; iii++) {
                if (iii <= 27 || iii >= 756 || iii % 28 == 0 || iii % 28 == 27) {
                    weights[i][ii].push(sizeInit(sizeInitTrend, 0.005))
                } else{
                    weights[i][ii].push(sizeInit(sizeInitTrend, sizeInitFactor))
                }
            }
        } else {
            for (let iii = 0; iii < 28*28; iii++) {
                weights[i][ii].push(sizeInit(sizeInitTrend, sizeInitFactor))
            }           
        }
    }
}
// let weights = [ //weights is a 3D array, D1 is the layers, D2 is nodes, D3 is the weights from this node to all the node of the previous layer, this is the original fixed version to generate the weights array, random within a range for all value.
//     Array.from({length:Node}, () => Array.from({length:28*28}, () => sizeInit(sizeInitTrend, sizeInitFactor))),
//     Array.from({length:Node}, () => Array.from({length:Node}, () => sizeInit(sizeInitTrend, sizeInitFactor))),
//     Array.from({length:10},  () => Array.from({length:Node}, () => sizeInit(sizeInitTrend, sizeInitFactor)))
// ]
let biases = [ //2D array, D1 is the layer, D2 is the biases(of nodes)
    Array.from({length: Node}, () => sizeInit(sizeInitTrend, sizeInitFactor)),
    Array.from({length: Node}, () => sizeInit(sizeInitTrend, sizeInitFactor)),
    Array.from({length: 10},  () => sizeInit(sizeInitTrend, sizeInitFactor))
];



function relu(x) {
    return Math.max(0, x);
}
function Drelu(x) { //Derivative of relu function
    return x > 0 ? 1 : 0;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function Dsigmoid(z) {    //Derivative of sigmoid function 
    const a = sigmoid(z);
    return a * (1 - a);
}

function saveModel(weights, biases) {//Save weights and biases
    fs.writeFileSync(weightsOutput, JSON.stringify(weights) + "\n", "utf8");
    fs.writeFileSync(biasesOutput, JSON.stringify(biases) + "\n", "utf8");
    console.log(`Model saved to ${weightsOutput} and ${biasesOutput}`);
}

function run(image, weights, biases) { //Calculate the activation and raw sum of a given sample picture
    let NN = []
    let NNZ = []
    let data = image
    let layer
    let z
    let activation_result
    function activation(PrevLayer, weight, bias, last) {// the function that calculate the activation based on given layer and previous layer, "last" parameter of boolean，only used by function "run"
        let node = []
        let z = []
        for (let i = 0; i < bias.length; i++){ //Looping through all the node in the current layer
            let sum = 0        
            for (let ii = 0; ii < PrevLayer.length; ii++){ //Looping through all the previous nodes to get the index of weights
                sum += PrevLayer[ii] * weight[i][ii]
            }
            sum += bias[i]
            z.push(sum)
            if (method === "relu"){
                if (last === true){ //using relu for the hiden layer and sigmoid for the last layer
                    node.push(sigmoid(sum))
                }else if(last === false){
                    node.push(relu(sum))
                }
            } else if (method === "sigmoid") {
                node.push(sigmoid(sum)) 
            }
            
        }
        return [node, z]
    }
    for (let i = 0; i < weights.length; i++) {
        if (i === weights.length - 1) {
            activation_result = activation(data, weights[i], biases[i], true)
            layer = activation_result[0]
            z = activation_result[1]
            NN.push(layer)
            NNZ.push(z)
            data = layer
        }else {
            activation_result = activation(data, weights[i], biases[i], false)
            layer = activation_result[0]
            z = activation_result[1]
            NN.push(layer)
            NNZ.push(z)
            data = layer
        }
        
    }
    return [NN, NNZ]
}

//we use delta to adjust bias and gradient to adjust weights
function train(images, labels, weights, biases) {
    let run_result
    let NN, NNZ
    let NextLayerErrorSum
    let delta = 0
    let deltaL = []
    let deltaNext = []
    let gradient
    for (let i = 0; i < images.length; i++) { //Loop through all the sample
        run_result = run(images[i], weights, biases)
        NN = run_result[0] //2D array, D1 is the layers, D2 is the activations(nodes)
        NNZ = run_result[1] //2D array, D1 is the layers, D2 is the z(activation that doesn't include sigmoid or relu)
        deltaL = [] //Refresh for next image
        deltaNext = [] //Refresh for next image
        gradient = 0
        for (let ii = biases.length - 1; ii >= 0 ; ii--) { //Loop through all the layer of the NN backwards, index of the layer is ii
            for (let iii = 0; iii < biases[ii].length; iii++) { // Loop through all the nodes in the layer ii, index of the node is iii
                NextLayerErrorSum = 0 //Refresh for next layer
                delta = 0 //Refresh for next layer
                if (ii === biases.length - 1) { //check for output layer
                    if (iii === labels[i]) {
                        delta = 2*(NN[ii][iii]- 1) * Dsigmoid(NNZ[ii][iii]) //formula {δ[2]=2(a[2]−y)⊙σ′(z[2])}   ∂a/∂z = σ′(z) so I guess we are putting z through Dsigmoid
                        deltaNext.push(delta) //this order is fine since we are propagating forwar in node and only backwards in layers
                    } else {
                        delta = 2*(NN[ii][iii]- 0) * Dsigmoid(NNZ[ii][iii]) //y is the either 1 or 0, 1 for correct 0 for wrong
                        deltaNext.push(delta)
                    }
                    // const y = (iii === labels[i]) ? 1 : 0;
                    // delta = NN[ii][iii] - y;  // sigmoid + cross-entropy
                    // deltaNext.push(delta)
                    biases[ii][iii] = biases[ii][iii] - (step * delta) //update the bias, bias_original - η*(delta)

                    for (let iiii = 0; iiii < NN[ii-1].length; iiii++) { //Loop through all the nodes in layer "ii-1", with iiii as the index of the node
                        gradient = delta * NN[ii-1][iiii] //gradient = ∂w/∂C = δ[2](a[1])
                        weights[ii][iii][iiii] = weights[ii][iii][iiii] - (step * gradient) //new_weight = old_weight − η⋅(∂w/∂C)​           updating the weight
                    }
                } else { //if not output layer
                    delta = 0
                    for (let iiii = 0; iiii < biases[ii + 1].length; iiii++) { //Loop through all the node of the next/higher layer, iiii as the index of the node
                        NextLayerErrorSum += weights[ii + 1][iiii][iii] * deltaL[iiii]      /* δH1 = w[2]1,1 * δO1 + w[2]1,2 * δO2
                                                                                           δH1 = w[2]1,1 * δO1 + w[2]1,2 * δO2           
                                                                                           hidden layer (layer 1) has neuron H1, H2; 
                                                                                           next layer (layer 2) has neurons O1, O2; 
                                                                                           weight w[2]i,j mean "weight from hidden neuron Hj to output neuron Oj"
                                                                                           */
                    }
                    if (method === "sigmoid"){
                        delta = NextLayerErrorSum * Dsigmoid(NNZ[ii][iii]) //(W[l+1])Tδ[l+1 * σ′(z[l])
                    } else if (method === "relu"){
                        delta = NextLayerErrorSum * Drelu(NNZ[ii][iii]) //(W[l+1])Tδ[l+1 * σ′(z[l])
                    }
                    deltaNext.push(delta)
                    biases[ii][iii] = biases[ii][iii] - (step * delta) //update the bias, bias_original - η*(delta)

                    for (let iiii = 0; iiii < NN[ii].length; iiii++) { //Loop through all the nodes in layer "ii", with iiii as the index of the node           We can use ii instead of ii-1 now because the length of the array as been increased, everything is shifted
                        gradient = delta * NN[ii][iiii] //gradient = ∂w/∂C = δ[2](a[1])
                        weights[ii][iii][iiii] = weights[ii][iii][iiii] - (step * gradient) //new_weight = old_weight − η⋅(∂w/∂C)​           updating the weight
                    }
                }
            }
            if (ii === biases.length - 1) {
                NN.unshift(images[i])
            }
            deltaL = structuredClone(deltaNext)
            deltaNext = []
        }
        //ETA calculator
        let elapsed = (Date.now() - startTime) / 1000; // seconds
        let avgPerSample = elapsed / (i + 1);
        let remaining = images.length - (i + 1);
        let eta = avgPerSample * remaining;

        console.log(
        `Sample ${i+1}/${images.length} | elapsed: ${elapsed.toFixed(1)}s | ETA: ${eta.toFixed(1)}s`
        );
        //ETA calculator
    }
}


train(images, labels, weights, biases);
saveModel(weights, biases);