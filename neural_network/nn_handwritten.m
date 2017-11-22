data = [];
training_data_ai = [];
test_data_ai = [];
training_data_la = [];
test_data_la = [];
training_data_ta = [];
test_data_ta = [];

%% Number of clusters.
num_of_clusters = 5;

 % Read data.
data_ai = dlmread('../data/hand_written/features/ai.ldf');
frame_ai = data_ai(:,1);
num_frames_ai = size(data_ai,1);
data_ai(:,1) = [];

data_la = dlmread('../data/hand_written/features/lA.ldf');
frame_la = data_la(:,1);
num_frames_la = size(data_la,1);
data_la(:,1) = [];

data_ta = dlmread('../data/hand_written/features/tA.ldf');
frame_ta = data_ta(:,1);
num_frames_ta = size(data_ta,1);
data_ta(:,1) = [];

% Set ai data.
for i=1:num_frames_ai
    j=1;
    k=1;
    temp = [];
    run = true;
    while (j<size(data_ai,2) && run == true)
        if(data_ai(i,j) == 0)
            run = false;
        else
            temp(k,1) = data_ai(i,j);
            temp(k,2) = data_ai(i,j+1);
            k = k+1;
            j = j+2;
        end   
        
    end
    if(i<=uint16(0.7*num_frames_ai))
        training_data_ai = [training_data_ai;temp];
    else
        test_data_ai = [test_data_ai;temp];
    end
    
end

% Set la data.
for i=1:num_frames_la
    j=1;
    k=1;
    temp = [];
    run = true;
    while (j<size(data_la,2) && run == true)
        if(data_la(i,j) == 0)
            run = false;
        else
            temp(k,1) = data_la(i,j);
            temp(k,2) = data_la(i,j+1);
            k = k+1;
            j = j+2;
        end   
    end
    if(i<=uint16(0.7*num_frames_la))
        training_data_la = [training_data_la;temp];
    else
        test_data_la = [test_data_la;temp];
    end
    
end

% Set ta data.
for i=1:size(data_ta,1)
    j=1;
    k=1;
    temp = [];
    run = true;
    while (j<size(data_ta,2) && run == true)
        if(data_ta(i,j) == 0)
            run = false;
        else
            temp(k,1) = data_ta(i,j);
            temp(k,2) = data_ta(i,j+1);
            k = k+1;
            j = j+2;
        end   

    end
    if(i<=uint16(0.7*num_frames_ta))
        training_data_ta = [training_data_ta;temp];
    else
        test_data_ta = [test_data_ta;temp];
    end
    
end

%% Apply Feature Extraction on training and testing data.

% Ai features
total_data_ai = [training_data_ai;test_data_ai];
first_derivate_ai = first_derivative(total_data_ai);
second_derivative_ai = second_derivative(first_derivate_ai);
curvature_ai = curvature(first_derivate_ai,second_derivative_ai);
total_features_ai = [total_data_ai(5:size(total_data_ai,1)-4,:) first_derivate_ai(3:size(first_derivate_ai,1)-2,:) second_derivative_ai curvature_ai];      
training_features_ai = total_features_ai((1:size(training_data_ai,1)-4),:);
testing_features_ai = total_features_ai((size(training_features_ai,1)+1:size(total_features_ai,1)),:);

% La features
total_data_la = [training_data_la;test_data_la];
first_derivate_la = first_derivative(total_data_la);
second_derivative_la = second_derivative(first_derivate_la);
curvature_la = curvature(first_derivate_la,second_derivative_la);
total_features_la = [total_data_la(5:size(total_data_la,1)-4,:) first_derivate_la(3:size(first_derivate_la,1)-2,:) second_derivative_la curvature_la];      
training_features_la = total_features_la((1:size(training_data_la,1)-4),:);
testing_features_la = total_features_la((size(training_features_la,1)+1:size(total_features_la,1)),:);

% Ta features
total_data_ta = [training_data_ta;test_data_ta];
first_derivate_ta = first_derivative(total_data_ta);
second_derivative_ta = second_derivative(first_derivate_ta);
curvature_ta = curvature(first_derivate_ta,second_derivative_ta);
total_features_ta = [total_data_ta(5:size(total_data_ta,1)-4,:) first_derivate_ta(3:size(first_derivate_ta,1)-2,:) second_derivative_ta curvature_ta];      
training_features_ta = total_features_ta((1:size(training_data_ta,1)-4),:);
testing_features_ta = total_features_ta((size(training_features_ta,1)+1:size(total_features_ta,1)),:);


%% Input to neural network
input_neural_network = [total_features_ai;total_features_la;total_features_ta];

%% Target to neural network
target = zeros(3,size(input_neural_network,1));  
target(1,1:size(total_features_ai,1)) = 1;
target(2,size(total_features_ai,1) + 1:size(total_features_ai,1) + size(total_features_la,1)) = 1;
target(3,size(total_features_ai,1) + size(total_features_la,1) + 1:size(input_neural_network,1)) = 1;
target = target';

%% Neural network start
nnstart;




