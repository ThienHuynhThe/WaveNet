net = layerGraph();

tempLayers = [
    imageInputLayer([64 64 9],"Name","input")
    convolution2dLayer([3 3],32,"Name","conv","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")];
net = addLayers(net,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","conv_1","Padding","same")
    groupedConvolution2dLayer([3 3],2,16,"Name","groupedconv_1","Padding","same")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],16,"Name","conv_2","Padding","same")
    groupedConvolution2dLayer([3 3],2,16,"Name","groupedconv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_2")];
net = addLayers(net,tempLayers);

tempLayers = additionLayer(2,"Name","addition");
net = addLayers(net,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","conv_3","Padding","same")
    groupedConvolution2dLayer([3 3],2,16,"Name","groupedconv_3","Padding","same")
    reluLayer("Name","relu_3")
    convolution2dLayer([1 1],16,"Name","conv_4","Padding","same")
    groupedConvolution2dLayer([3 3],2,16,"Name","groupedconv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_4")];
net = addLayers(net,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
net = addLayers(net,tempLayers);

tempLayers = [
    averagePooling2dLayer([1 32],"Name","avgpool2d")
    convolution2dLayer([32 1],4,"Name","conv_5")];
net = addLayers(net,tempLayers);

tempLayers = [
    averagePooling2dLayer([32 1],"Name","avgpool2d_1")
    convolution2dLayer([1 32],4,"Name","conv_6")];
net = addLayers(net,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat");
net = addLayers(net,tempLayers);

tempLayers = sigmoidLayer("Name","sigmoid");
net = addLayers(net,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication")
    convolution2dLayer([1 1],32,"Name","conv_7","Padding","same")];
net = addLayers(net,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_1")
    groupedConvolution2dLayer([3 3],2,32,"Name","groupedconv_5","Padding","same","Stride",[2 2])];
net = addLayers(net,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","conv_8","Padding","same")
    groupedConvolution2dLayer([3 3],4,16,"Name","groupedconv_6","Padding","same")
    reluLayer("Name","relu_5")
    convolution2dLayer([1 1],16,"Name","conv_9","Padding","same")
    groupedConvolution2dLayer([3 3],4,16,"Name","groupedconv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_6")];
net = addLayers(net,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2");
net = addLayers(net,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","conv_10","Padding","same")
    groupedConvolution2dLayer([3 3],4,16,"Name","groupedconv_8","Padding","same")
    reluLayer("Name","relu_7")
    convolution2dLayer([1 1],16,"Name","conv_11","Padding","same")
    groupedConvolution2dLayer([3 3],4,16,"Name","groupedconv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_8")];
net = addLayers(net,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3");
net = addLayers(net,tempLayers);

tempLayers = [
    averagePooling2dLayer([1 16],"Name","avgpool2d_2")
    convolution2dLayer([16 1],4,"Name","conv_12")];
net = addLayers(net,tempLayers);

tempLayers = [
    averagePooling2dLayer([16 1],"Name","avgpool2d_3")
    convolution2dLayer([1 16],4,"Name","conv_13")];
net = addLayers(net,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat_1");
net = addLayers(net,tempLayers);

tempLayers = sigmoidLayer("Name","sigmoid_1");
net = addLayers(net,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_2")
    convolution2dLayer([1 1],64,"Name","conv_14","Padding","same")];
net = addLayers(net,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_3")
    groupedConvolution2dLayer([3 3],2,64,"Name","groupedconv_10","Padding","same","Stride",[2 2])];
net = addLayers(net,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","conv_15","Padding","same")
    groupedConvolution2dLayer([3 3],8,16,"Name","groupedconv_11","Padding","same")
    reluLayer("Name","relu_9")
    convolution2dLayer([1 1],16,"Name","conv_16","Padding","same")
    groupedConvolution2dLayer([3 3],8,16,"Name","groupedconv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_10")];
net = addLayers(net,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4");
net = addLayers(net,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","conv_17","Padding","same")
    groupedConvolution2dLayer([3 3],8,16,"Name","groupedconv_13","Padding","same")
    reluLayer("Name","relu_11")
    convolution2dLayer([1 1],16,"Name","conv_18","Padding","same")
    groupedConvolution2dLayer([3 3],8,16,"Name","groupedconv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_12")];
net = addLayers(net,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5");
net = addLayers(net,tempLayers);

tempLayers = [
    averagePooling2dLayer([1 8],"Name","avgpool2d_4")
    convolution2dLayer([8 1],4,"Name","conv_19")];
net = addLayers(net,tempLayers);

tempLayers = [
    averagePooling2dLayer([8 1],"Name","avgpool2d_5")
    convolution2dLayer([1 8],4,"Name","conv_20")];
net = addLayers(net,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat_2");
net = addLayers(net,tempLayers);

tempLayers = sigmoidLayer("Name","sigmoid_2");
net = addLayers(net,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_4")
    convolution2dLayer([1 1],128,"Name","conv_21","Padding","same")];
net = addLayers(net,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_5")
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(32,"Name","fc")
    dropoutLayer(0.3,"Name","dropout")
    fullyConnectedLayer(8,"Name","fc_1")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
net = addLayers(net,tempLayers);

net = connectLayers(net,"relu","conv_1");
net = connectLayers(net,"relu","addition/in2");
net = connectLayers(net,"relu_2","addition/in1");
net = connectLayers(net,"addition","conv_3");
net = connectLayers(net,"addition","addition_1/in2");
net = connectLayers(net,"relu_4","addition_1/in1");
net = connectLayers(net,"addition_1","avgpool2d");
net = connectLayers(net,"addition_1","avgpool2d_1");
net = connectLayers(net,"addition_1","multiplication_1/in1");
net = connectLayers(net,"conv_5","depthcat/in1");
net = connectLayers(net,"conv_6","depthcat/in2");
net = connectLayers(net,"depthcat","sigmoid");
net = connectLayers(net,"depthcat","multiplication/in2");
net = connectLayers(net,"sigmoid","multiplication/in1");
net = connectLayers(net,"conv_7","multiplication_1/in2");
net = connectLayers(net,"groupedconv_5","conv_8");
net = connectLayers(net,"groupedconv_5","addition_2/in2");
net = connectLayers(net,"relu_6","addition_2/in1");
net = connectLayers(net,"addition_2","conv_10");
net = connectLayers(net,"addition_2","addition_3/in2");
net = connectLayers(net,"relu_8","addition_3/in1");
net = connectLayers(net,"addition_3","avgpool2d_2");
net = connectLayers(net,"addition_3","avgpool2d_3");
net = connectLayers(net,"addition_3","multiplication_3/in1");
net = connectLayers(net,"conv_12","depthcat_1/in1");
net = connectLayers(net,"conv_13","depthcat_1/in2");
net = connectLayers(net,"depthcat_1","sigmoid_1");
net = connectLayers(net,"depthcat_1","multiplication_2/in2");
net = connectLayers(net,"sigmoid_1","multiplication_2/in1");
net = connectLayers(net,"conv_14","multiplication_3/in2");
net = connectLayers(net,"groupedconv_10","conv_15");
net = connectLayers(net,"groupedconv_10","addition_4/in2");
net = connectLayers(net,"relu_10","addition_4/in1");
net = connectLayers(net,"addition_4","conv_17");
net = connectLayers(net,"addition_4","addition_5/in2");
net = connectLayers(net,"relu_12","addition_5/in1");
net = connectLayers(net,"addition_5","avgpool2d_4");
net = connectLayers(net,"addition_5","avgpool2d_5");
net = connectLayers(net,"addition_5","multiplication_5/in1");
net = connectLayers(net,"conv_19","depthcat_2/in1");
net = connectLayers(net,"conv_20","depthcat_2/in2");
net = connectLayers(net,"depthcat_2","sigmoid_2");
net = connectLayers(net,"depthcat_2","multiplication_4/in2");
net = connectLayers(net,"sigmoid_2","multiplication_4/in1");
net = connectLayers(net,"conv_21","multiplication_5/in2");