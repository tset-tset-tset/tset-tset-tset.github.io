function setup(){
    createCanvas(256, 256);
    stroke(0);
    background(255);
    pixelDensity(1);
    text('Now loading...', 85, 128);
    strokeWeight(5);
    pix2pixModel = null;
    readyTransfer = false;
    //tf.loadGraphModel('http://127.0.0.1:8000/pix2pix/tfjs_graph_model/model.json').then(model => {
    tf.loadGraphModel('https://tset-tset-tset.github.io/pix2pix/generator_tfjs_graph_model/model.json').then(model => {
        tf.tidy(() => {
            model.predict(tf.zeros([1, 256, 256, 3]));
        });
        pix2pixModel = model;
        readyTransfer = true;
        background(255);
    })
    select('#clear_canvas').mousePressed(() => {
        background(255);
    });
    select('#transfer').mousePressed(() => {
        pix2pixTransfer();
    });
}

function draw(){
    if (mouseIsPressed && readyTransfer) {
        line(mouseX, mouseY, pmouseX, pmouseY);
    }
}

function pix2pixTransfer(){
    if (!readyTransfer) {
        return;
    }
    resultTensor = tf.tidy(() => {
        // tf.browser.fromPixels() returns a Tensor from an canvas element.
        let inputTensor = tf.browser.fromPixels(select('canvas').elt).toFloat();

        // normalize [0 ~ 255] -> [-1 ~ 1]
        inputTensor = inputTensor.div(tf.scalar(127.5)).sub(tf.scalar(1.0));

        // expand dimension (HWC -> NHWC)
        inputTensor = inputTensor.expandDims();

        // make a prediction through pix2pix
        let resultTensor = pix2pixModel.predict(inputTensor);

        // to browser [-1 ~ 1] -> [0 ~ 1]
        resultTensor = resultTensor.div(tf.scalar(2)).add(tf.scalar(0.5));

        // squeeze dimension (NHWC -> HWC)
        return resultTensor.squeeze();
    });

    tf.browser.toPixels(resultTensor).then(data => {
        select('#output').html('');
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 256;
        const context = canvas.getContext('2d');
        const imageData = context.createImageData(256, 256);
        data.forEach((value, index) => { imageData.data[index] = value });
        context.putImageData(imageData, 0, 0);
        createImg(canvas.toDataURL(), "transfer result").parent('output');
    });
    console.log(tf.memory());
    readyTransfer = true;
}
