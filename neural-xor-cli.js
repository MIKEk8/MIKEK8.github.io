// Neural XOR Trainer - Console Version

var learningRate = 2;
var maxIterations = 5000;

function randomWeight() {
    return (Math.random() - 0.5) * 2;  // [-1, 1]
}

// 1 hidden neuron + skip connections from inputs to output
var hidden = { i1: randomWeight(), i2: randomWeight(), b: randomWeight() };
var outWeights = {
    h: randomWeight(),   // from hidden
    i1: randomWeight(),  // skip connection from input 1
    i2: randomWeight(),  // skip connection from input 2
    b: randomWeight()
};

var sigmoid = (x) => 1 / (1 + Math.exp(-x));
var sigmoidDerivative = (fx) => fx * (1 - fx);

var trainingData = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];

function forward(i1, i2) {
    var hSum = i1 * hidden.i1 + i2 * hidden.i2 + hidden.b;
    var hOut = sigmoid(hSum);

    // Output receives: hidden output + direct inputs (skip connections)
    var outSum = hOut * outWeights.h + i1 * outWeights.i1 + i2 * outWeights.i2 + outWeights.b;
    var output = sigmoid(outSum);

    return { hOut, output };
}

function getPredictions() {
    var cases = [[0, 0], [0, 1], [1, 0], [1, 1]];
    return cases.map(([i1, i2]) => {
        var result = forward(i1, i2);
        var expected = i1 ^ i2;
        var error = Math.abs(expected - result.output);
        return { i1, i2, expected, output: result.output, error, correct: error < 0.1 };
    });
}

function learn() {
    // Accumulate gradients
    var hGrads = { i1: 0, i2: 0, b: 0 };
    var outGrads = { h: 0, i1: 0, i2: 0, b: 0 };

    for (var t = 0; t < trainingData.length; t++) {
        var [i1, i2, expected] = trainingData[t];
        var { hOut, output } = forward(i1, i2);

        var outputError = expected - output;
        var outputDelta = sigmoidDerivative(output) * outputError;

        // Output layer gradients (including skip connections)
        outGrads.b += outputDelta;
        outGrads.h += outputDelta * hOut;
        outGrads.i1 += outputDelta * i1;  // skip connection gradient
        outGrads.i2 += outputDelta * i2;  // skip connection gradient

        // Hidden layer gradients
        var hError = outputDelta * outWeights.h;
        var hDelta = sigmoidDerivative(hOut) * hError;
        hGrads.b += hDelta;
        hGrads.i1 += hDelta * i1;
        hGrads.i2 += hDelta * i2;
    }

    // Apply gradients
    outWeights.b += learningRate * outGrads.b;
    outWeights.h += learningRate * outGrads.h;
    outWeights.i1 += learningRate * outGrads.i1;
    outWeights.i2 += learningRate * outGrads.i2;
    hidden.b += learningRate * hGrads.b;
    hidden.i1 += learningRate * hGrads.i1;
    hidden.i2 += learningRate * hGrads.i2;

    var predictions = getPredictions();
    return predictions.every(p => p.error < 0.1);
}

function printPredictions() {
    console.log('\nПредсказания XOR:');
    console.log('Вход\tОжид.\tВыход\t\tОшибка');
    getPredictions().forEach(p => {
        var status = p.correct ? '✓' : '✗';
        console.log(`${p.i1}^${p.i2}\t${p.expected}\t${p.output.toFixed(4)}\t\t${p.error.toFixed(4)} ${status}`);
    });
}

function printWeights() {
    console.log('\nВеса сети:');
    console.log('Hidden:', hidden);
    console.log('Output:', outWeights);
}

// Main training loop
console.log('=== Neural XOR Trainer ===');
console.log(`Learning rate: ${learningRate}, Max iterations: ${maxIterations}`);
console.log('Architecture: 1 hidden + skip connections (i1, i2 → output)');
console.log('\nНачальные веса:');
printWeights();
printPredictions();

var step = 0;
var converged = false;

for (var i = 0; i < maxIterations; i++) {
    step++;
    if (learn()) {
        converged = true;
        break;
    }

    // Print progress every 500 iterations
    if (step % 500 === 0) {
        console.log(`\n--- Итерация ${step} ---`);
        printPredictions();
    }
}

console.log('\n=== РЕЗУЛЬТАТ ===');
console.log(`Итераций: ${step}`);
console.log(`Статус: ${converged ? 'СОШЛОСЬ!' : 'НЕ СОШЛОСЬ'}`);
printWeights();
printPredictions();
