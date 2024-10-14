const classes = {
    0: "ants",
    1: "bees",
    2: 'beetle',
    3: 'catterpillar',
    4: 'earthworms',
    5: 'earwig',
    6: 'grasshopper',
    7: 'moth',
    8: 'slug',
    9: 'snail',
    10: 'wasp',
    11: 'weevil'
};

let model;  // Global model variable

async function loadModel() {
    // Load the TFLite model
    model = await tflite.loadTFLiteModel('models/pest_classifier.tflite');
    console.log('Model Loaded Successfully');
}

// Image reading function
function readImage(input) {
    if (input.files && input.files[0]) {
        let reader = new FileReader();

        reader.onload = function (e) {
            document.getElementById('selectedImage').src = e.target.result;
        };

        reader.readAsDataURL(input.files[0]);
    }
}

// Event listener for image upload
document.getElementById('imageInput').addEventListener('change', function () {
    readImage(this);
});

function classifyImage() {
    if (!model) {
        alert('Model is not loaded yet. Please wait.');
        return;
    }

    // Preprocess image
    const image = document.getElementById('selectedImage');

    let tensor = tf.browser.fromPixels(image)
    tensor = tf.image.resizeBilinear(tensor, [1, 1]); // image size needs to be same as model inputs
    tensor = tf.expandDims(tensor)

    try {
        // Predict using the model (synchronously)
        tensor = tf.cast(tensor, "int32");
        const output = model.predict(tensor);

        // const output_max = tf.max(output.arraySync()[0]);
        output_val = tf.cast(output, 'float32')
        output_array = tf.softmax(output_val);
        // const maxIndex = output_array.argMax(-1).arraySync();
        console.log(tf.softmax(output_val).arraySync());
        console.log(output_array.max().arraySync());

        const output_values = tf.softmax(output.arraySync()[0]);
        console.log(classes[output_values.argMax().arraySync()]);
        const predictedClassName = classes[output_values.argMax().arraySync()];
        document.getElementById('result').textContent = `Predicted Class: ${predictedClassName}`;

    } catch (err) {
        console.error("Prediction error: ", err);
    }
}

// Load the model as soon as the page loads
window.onload = function () {
    loadModel();
};