load mehamn_data.txt;

%assign training set for Y
data_output = (mehamn_data(:,9));


figure
plot(data_output)
xlabel("time")
ylabel("height")
title("Height")
1
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
2

numTimeStepsTrain = floor(0.9*numel(dataInput_i(:,1)));
dataTrain = dataInput_i([1:numTimeStepsTrain+1],:);
dataTrain_Y = data_output(1:numTimeStepsTrain+1);
dataTest = dataInput_i([numTimeStepsTrain+1:end],:);

mu = mean(dataTrain_Y);
sig = std(dataTrain_Y);
dataTrainStandardized_Y = (dataTrain_Y - mu) / sig;


for i = 1:15
    mu(1,i) = mean(dataTrain(:,i));
    sig(1,i) = std(dataTrain(:,i));  
end
   
for i = 1:15
dataTrainStandardized_f(:,i) = (dataTrain(:,i) - mu(1,i)) / sig(1,i);
end

figure
plot(dataTrainStandardized_f)
title('dataTrainStandardized_f')
3

XTrain = dataTrainStandardized_f([1:end-8],:);


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

for i = 1:15
mu(1,i) = mean(dataTest(:,1));
sig(1,i) = std(dataTest(:,1));
dataTestStandardized(:,i) = (dataTest(:,i) - mu(1,i))/sig(1,i);
end

XTest = dataTestStandardized([1:end-8],:);
X = transpose(XTest);

net = predictAndUpdateState(net,X1);
[net,YPred] = predictAndUpdateState(net,Y(end));

numTimeStepsTest = numel(X(:,1));

for i = 9:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-8),'ExecutionEnvironment','gpu');
end
0
YPred = sig*YPred + mu;
YTest = dataTest(9:end);
rmse = sqrt(mean((YPred-YTest).^2))
1

figure
plot(dataTrain_Y(1:end-8))
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

fprintf('press any key \n')
pause

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)