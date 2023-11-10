clear

% Set the base directory to the current directory
baseDir = pwd;

trainFilePath = fullfile(baseDir, 'commonvoice', 'train', 'train.tsv');
valFilePath = fullfile(baseDir, 'commonvoice', 'validation', 'validation.tsv');

trainTable = readtable(trainFilePath, FileType="text", Delimiter="tab");
valTable = readtable(valFilePath, FileType="text", Delimiter="tab");
dataTable = [trainTable; valTable];

% Sort speakers by how many files they speak on
dataTable.client_id = string(dataTable.client_id);
dataTable.path = string(dataTable.path);
ids = unique(dataTable.client_id);
numIds = length(ids);
counts = zeros(numIds, 1);
for i = 1:length(ids)
    counts(i) = sum(strcmp(dataTable.client_id,ids(i)));
end
[s, idxs] = sort(counts);

% Take speakers with around 14-22 files
assert(s(743) == 14 && s(752) == 22);
idxs = idxs(743:752);
ids = ids(idxs);
rows = ismember(dataTable.client_id,ids);

% Use the relative path for the 'train' and 'validation' directories
trainClipsDir = fullfile(baseDir, 'commonvoice', 'train', 'clips');
valClipsDir = fullfile(baseDir, 'commonvoice', 'validation', 'clips');

% Get paths for each file in dataTable
trainPaths = repmat({trainClipsDir}, height(trainTable), 1);
valPaths = repmat({valClipsDir}, height(valTable), 1);
paths = [trainPaths; valPaths]; % Concatenates the cell arrays of paths

% Only take paths for selected files
files = fullfile(baseDir, 'commonvoice', 'train', 'clips', string(dataTable.path(rows))) + ".wav";


% Get speaker IDs, create datastore, and assign speaker labels as 1-10
speakers = string(dataTable.client_id(rows));
ads = audioDatastore(files);
ads.Labels = categorical(speakers,unique(speakers),string(1:length(unique(speakers))));
[adsTrain,adsTest] = splitEachLabel(ads,0.8);
adsTrain
trainDatastoreCount = countEachLabel(adsTrain)
adsTest
testDatastoreCount = countEachLabel(adsTest)
[sampleTrain,dsInfo] = read(adsTrain);
sound(sampleTrain,dsInfo.SampleRate)
reset(adsTrain)
fs = dsInfo.SampleRate;
windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);
afe = audioFeatureExtractor(SampleRate=fs, ...
    Window=hamming(windowLength,"periodic"),OverlapLength=overlapLength, ...
    zerocrossrate=true,shortTimeEnergy=true,pitch=true,mfcc=true);
featureMap = info(afe)
features = [];
labels = [];
energyThreshold = 0.005;
zcrThreshold = 0.2;

allFeatures = extract(afe,adsTrain);
allLabels = adsTrain.Labels;

for ii = 1:numel(allFeatures)

    thisFeature = allFeatures{ii};

    isSpeech = thisFeature(:,featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = thisFeature(:,featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    thisFeature(~voicedSpeech,:) = [];
    thisFeature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    label = repelem(allLabels(ii),size(thisFeature,1));
    
    features = [features;thisFeature];
    labels = [labels,label];
end
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;
% Convert labels to categorical if they aren't already
labels = categorical(labels);

% Split data into training and test sets 
c = cvpartition(labels, 'HoldOut', 0.2); % holding 20% of data for testing
trainingIndices = training(c);
testIndices = test(c);

% Train the SVM model
SVMModel = fitcecoc(features(trainingIndices, :), labels(trainingIndices));

% Evaluate the classifier on the training data
trainPrediction = predict(SVMModel, features(trainingIndices, :));
confMatTrain = confusionmat(labels(trainingIndices), trainPrediction);
trainAccuracy = sum(diag(confMatTrain)) / sum(confMatTrain, 'all');
fprintf('\nTraining accuracy = %.2f%%\n', trainAccuracy * 100);
% Evaluate the classifier on the test data
testPrediction = predict(SVMModel, features(testIndices, :));
confMatTest = confusionmat(labels(testIndices), testPrediction);
testAccuracy = sum(diag(confMatTest)) / sum(confMatTest, 'all');
fprintf('\nTest accuracy = %.2f%%\n', testAccuracy * 100);
% Confusion chart for the test data
figure(Units="normalized", Position=[0.4 0.4 0.4 0.4])
confusionchart(labels(testIndices), testPrediction, ...
    'Title', 'Confusion Matrix for Test Data', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
features = [];
labels = [];
numVectorsPerFile = [];

allFeatures = extract(afe,adsTest);
allLabels = adsTest.Labels;

for ii = 1:numel(allFeatures)

    thisFeature = allFeatures{ii};

    isSpeech = thisFeature(:,featureMap.shortTimeEnergy) > energyThreshold;
    isVoiced = thisFeature(:,featureMap.zerocrossrate) < zcrThreshold;

    voicedSpeech = isSpeech & isVoiced;

    thisFeature(~voicedSpeech,:) = [];
    numVec = size(thisFeature,1);
    thisFeature(:,[featureMap.zerocrossrate,featureMap.shortTimeEnergy]) = [];
    
    label = repelem(allLabels(ii),numVec);
    
    numVectorsPerFile = [numVectorsPerFile,numVec];
    features = [features;thisFeature];
    labels = [labels,label];
end
features = (features-M)./S;
% Predict using the SVM model
prediction = predict(SVMModel, features);

% Convert predictions to categorical if they are not already
prediction = categorical(prediction);

% Evaluate the classifier on a per-frame basis
confMatPerFrame = confusionmat(labels(:), prediction);
figure(Units="normalized", Position=[0.4 0.4 0.4 0.4])
confusionchart(confMatPerFrame, ...
    'Title', 'Test Accuracy (Per Frame)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% Assuming C is your confusion matrix
C = confusionmat(labels, prediction);

% Compute overall accuracy
overallAccuracy = sum(diag(C)) / sum(C(:));

% Display the overall accuracy
fprintf('Overall Accuracy: %.2f%%\n', overallAccuracy * 100);
% Now let's calculate the accuracy per file
r2 = zeros(size(adsTest.Labels));
idx = 1;
for ii = 1:numel(adsTest.Files)
    r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
    idx = idx + numVectorsPerFile(ii);
end
r2 = categorical(r2);

% Evaluate the classifier on a per-file basis
confMatPerFile = confusionmat(adsTest.Labels, r2);
figure(Units="normalized", Position=[0.4 0.4 0.4 0.4])
confusionchart(confMatPerFile, ...
    'Title', 'Test Accuracy (Per File)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');