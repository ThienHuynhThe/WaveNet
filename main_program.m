clear
load WaveNet.m  % load WaveNet
% load and divide dataset into subset of training, validation, and test
imds = imageDatastore('radcomDatasetnew','IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.mat'});
[imdsTrain,imdsTest,imdsVal] = splitEachLabel(imds,0.7,0.15,'randomized');
imdsTrain.Labels    = categorical(imdsTrain.Labels);imdsTrain.ReadFcn = @readFcnMatFile;
imdsTest.Labels     = categorical(imdsTest.Labels);imdsTest.ReadFcn = @readFcnMatFile;
imdsVal.Labels      = categorical(imdsVal.Labels);imdsVal.ReadFcn = @readFcnMatFile;

% training options configuration
batchSize   = 128;
ValFre      = fix(length(imdsTrain.Files)/batchSize)
options = trainingOptions('adam', ...
    'MiniBatchSize',batchSize, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch',...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.2,...
    'ValidationData',imdsVal, ...
    'ValidationFrequency',ValFre, ...
    'ValidationPatience',8, ...
    'Verbose',true ,...
    'VerboseFrequency',ValFre,...
    'Plots','training-progress',...
    'ExecutionEnvironment','multi-gpu',...
    'OutputNetwork','best-validation-loss');

% train the model with the imdsTrain set and validate with the imdsVal set.
trainednet = trainNetwork(imdsTrain,lgraph,options);

% measure the classification accuracy of WaveNet on the test set
YPred = classify(trainednet,imdsTest,'MiniBatchSize',512,'ExecutionEnvironment','gpu');
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest)





