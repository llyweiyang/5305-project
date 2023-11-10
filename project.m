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

% Enhancement of data
originalSpeakers = string(dataTable.client_id(rows));
augmentedFiles = strings(size(files));
augmentedSpeakers = strings(size(originalSpeakers));
for i = 1:numel(files)
    [audioIn, fs] = audioread(files(i));
    stretchFactor = rand * 0.2 + 0.9; % Random stretch factor between 0.9 and 1.1
    audioOut = stretchAudio(audioIn, stretchFactor);
    augmentedFileName = strrep(files(i), '.wav', '_augmented.wav');
    audiowrite(augmentedFileName, audioOut, fs);
    augmentedFiles(i) = augmentedFileName;
    augmentedSpeakers(i) = originalSpeakers(i); % Assign the same speaker ID to the augmented file
end
combinedFiles = [files; augmentedFiles];
combinedSpeakers = [originalSpeakers; augmentedSpeakers];

% Create an audioDatastore with the combined files array
ads = audioDatastore(combinedFiles);

% Assign the combined speakers to the Labels of the datastore
ads.Labels = categorical(combinedSpeakers, unique(combinedSpeakers), string(1:length(unique(combinedSpeakers))));

% Continue with splitting into training and testing sets
[adsTrain,adsTest] = splitEachLabel(ads,0.8);

% Display the datastore and the number of speakers in the train datastore and test datastore.
adsTrain
trainDatastoreCount = countEachLabel(adsTrain)
adsTest
testDatastoreCount = countEachLabel(adsTest)

% Preview content and play
[sampleTrain,dsInfo] = read(adsTrain);
% sound(sampleTrain,dsInfo.SampleRate)
reset(adsTrain);

% Extract features from each frame
fs = dsInfo.SampleRate;
windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);
afe = audioFeatureExtractor(SampleRate=fs, ...
    Window=hamming(windowLength,"periodic"),OverlapLength=overlapLength, ...
    zerocrossrate=true,shortTimeEnergy=true,pitch=true,mfcc=true, ...
    harmonicRatio=true); % harmonicRatio improves accuracy
featureMap = info(afe)

% Extract features from the data set.
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

% Normalisation of features
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;

% Apply PCA for dimensionality reduction
% [coeff,score,~,~,explained] = pca(featuresNorm);

% Choose the number of components to keep by looking at the 'explained' variance
% numComponentsToKeep = find(cumsum(explained) >= 96, 1); % for example to keep 95% of variance
% featuresPca = score(:,1:numComponentsToKeep);

% Compute the KNN classifier
trainedClassifier = fitcknn(features,labels, ...
    Distance="euclidean", ...
    NumNeighbors=5, ...
    DistanceWeight="squaredinverse", ...
    Standardize=false, ...
    ClassNames=unique(labels));

% Perform cross-validation.
k = 5;
group = labels;
c = cvpartition(group,KFold=k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,CVPartition=c);

% Compute the validation accuracy.
validationAccuracy = 1 - kfoldLoss(partitionedModel,LossFun="ClassifError");
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

% Visualize the confusion chart.
validationPredictions = kfoldPredict(partitionedModel);
figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels,validationPredictions,title="Validation Accuracy", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

% test classifier
% Read files, extract features from the test set, and normalize them.
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

% Predict the label (speaker) for each frame
prediction = predict(trainedClassifier,features);
prediction = categorical(string(prediction));

% Visualize the confusion chart.
figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels(:),prediction,title="Test Accuracy (Per Frame)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
% Assuming C is your confusion matrix
C = confusionmat(labels, prediction);

% Compute overall accuracy
overallAccuracy = sum(diag(C)) / sum(C(:));

% Display the overall accuracy
fprintf('Overall Accuracy: %.2f%%\n', overallAccuracy * 100);

% Determine the mode of predictions for each file and then plot the confusion chart.
r2 = prediction(1:numel(adsTest.Files));
idx = 1;
for ii = 1:numel(adsTest.Files)
    r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
    idx = idx + numVectorsPerFile(ii);
end

figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(adsTest.Labels,r2,title="Test Accuracy (Per File)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

% Reference:Mathworks Documentation:Speaker Identification Using Pitch and MFCC.https://au.mathworks.com/help/audio/ug/speaker-identification-using-pitch-and-mfcc.html