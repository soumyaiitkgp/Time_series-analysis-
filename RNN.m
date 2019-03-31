load mehamn_data.txt;

%assign training set for Y
data_output = (mehamn_data(:,9));

figure
plot(data_output)
xlabel("time")
ylabel("height")
title("Height")

%---------------------------------------------------------------------------
%input set : 15 features
for i = 2:17
    D1 = (mehamn_data(:,i));
    dataInput(:,i-1) = (D1);
end

dataInput_i(:,[1:15]) = dataInput(:,[1:7 9:16]);
figure
plot(dataInput_i)
title("data_INPUT")

%split dataset-------------------------------------------------------------
numTimeStepsTrain = floor(0.9*numel(dataInput_i(:,1)));
dataTrain = dataInput_i([1:numTimeStepsTrain+1],:);
dataTrain_Y = data_output(1:numTimeStepsTrain+1);
dataTest = dataInput_i([numTimeStepsTrain+1:end],:);
dataTest_Y = data_output([numTimeStepsTrain+1:end],:);

%standardize the training dataset------------------------------------------
for i = 1:15
    mu(1,i) = mean(dataTrain(:,i));
    sig(1,i) = std(dataTrain(:,i));  
end
dataTrainStandardized_f = rand(157123,15); 

for i = 1:15
dataTrainStandardized_f(:,i) = (dataTrain(:,i) - mu(1,i)) / sig(1,i);
end

%standardize the output data set------------------------------------------
mu_Y = mean(dataTrain_Y);
sig_Y = std(dataTrain_Y);
dataTrainStandardized_Y = (dataTrain_Y - mu_Y) / sig_Y;

figure
plot(dataTrainStandardized_f)
title('dataTrainStandardized_f')
3

XTrain = dataTrain([1:end-8],:);
YTrain = dataTrainStandardized_Y(9:end);

%---------------------------training----------------------------------------

numFeatures = 15;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
        sequenceInputLayer(numFeatures)
    lstmLayer(100,'OutputMode','sequence')
    lstmLayer(100,'OutputMode','sequence')
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',2, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

X1 = transpose(XTrain);
Y = transpose(YTrain);
net = trainNetwork(X1,Y,layers,options);

%-------------------------------------------------------------


dataTestStandardized_f = rand(17459,15); 
for i = 1:15
dataTestStandardized_f(:,i) = (dataTest(:,i) - mu(1,i)) / sig(1,i);
end

XTest = dataTestStandardized_f([1:end-8],:);
X = transpose(XTest);

net = predictAndUpdateState(net,X1);

for i = 1:8
[net,YPred(:,i)] = predictAndUpdateState(net,X(:,end+i-8));
end

numTimeStepsTest = numel(X(1,:))+8;

for i = 1:numTimeStepsTest-8
    [net,YPred(:,i+8)] = predictAndUpdateState(net,X(:,i),'ExecutionEnvironment','gpu');
    
end

fprintf('key')
pause
1
Y_Pred = rand(1,17459);
Y_Pred(1,:) = YPred(1,:);
YPred = rand(1,17459);
YPred(1,:) = Y_Pred(1,:)

fprintf('key')
pause
YPred = sig_Y*YPred + mu_Y;
YTest = dataTest_Y(1:end);
rmse = sqrt(mean((YPred-YTest).^2))
fprintf('key')
pause

figure
plot(dataTrain_Y(1:end))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data_output(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

fprintf('key')
pause

YPred = transpose(YPred);

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")