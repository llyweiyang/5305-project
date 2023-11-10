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
mfccNames = arrayfun(@(i) sprintf('MFCC_%d', i), 1:13, 'UniformOutput', false); % Assuming you have 13 MFCCs
pitchName = {'Pitch'};
zcrName = {'ZeroCrossRate'};
steName = {'ShortTimeEnergy'};

predictorNames = {'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'Pitch', 'ZeroCrossRate'};
numTrees = 100; 
trainedClassifier = TreeBagger(numTrees, features, labels, ...
    'Method', 'classification', ...
    'OOBPrediction', 'On', ...
    'OOBVarImp', 'On', ...
    'MinLeafSize', 1, ... 
    'NumPredictorsToSample', 'all', ... 
    'PredictorSelection', 'curvature', ... 
    'PredictorNames', predictorNames); 
oobError = oobError(trainedClassifier);
fprintf('\nOut-of-bag classification error = %.2f%%\n', oobError(end) * 100);
% Get OOB predictions
oobPredictions = oobPredict(trainedClassifier);

% Convert OOB predictions to categorical if your labels are categorical
oobPredictionsCategorical = categorical(oobPredictions);

% Generate a confusion matrix using these OOB predictions
confusionMatrixOOB = confusionmat(labels, oobPredictionsCategorical);

% Visualize the confusion matrix
confusionchart(confusionMatrixOOB);
title('OOB Confusion Matrix');
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
prediction = trainedClassifier.predict(features);
prediction = categorical(prediction);
figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels(:),prediction,title="Test Accuracy (Per Frame)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
% Assuming C is your confusion matrix
C = confusionmat(labels, prediction);

% Compute overall accuracy
overallAccuracy = sum(diag(C)) / sum(C(:));

% Display the overall accuracy
fprintf('Overall Accuracy: %.2f%%\n', overallAccuracy * 100);
r2 = prediction(1:numel(adsTest.Files));
idx = 1;
for ii = 1:numel(adsTest.Files)
    r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
    idx = idx + numVectorsPerFile(ii);
end

figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(adsTest.Labels,r2,title="Test Accuracy (Per File)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");