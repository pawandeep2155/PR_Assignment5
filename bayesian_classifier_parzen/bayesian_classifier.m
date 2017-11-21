%% Get info for image data. 
street_data_info = dir('../data/image_data/features/street/*.jpg_color_edh_entropy');
coast_data_info = dir('../data/image_data/features/coast/*.jpg_color_edh_entropy');
forest_data_info = dir('../data/image_data/features/forest/*.jpg_color_edh_entropy');

%% Hyperparameter 'h' for gaussian kernel
h = 20;

%% Get data for each of the class. 
num_of_images_street = length(street_data_info);
num_of_images_coast = length(coast_data_info);
num_of_images_forest = length(forest_data_info);

data_street = zeros(36,23 * num_of_images_street);
data_coast = zeros(36,23 * num_of_images_street);
data_forest = zeros(36,23 * num_of_images_street);

for i=1:num_of_images_street % Get data for street class
   current_file_name = street_data_info(i).name;
   image_path = strcat('../data/image_data/features/street/',current_file_name);
   image_data = load(image_path);
   data_street(:,(i-1) * 23 + 1:(i-1) * 23 + 23) = image_data;
end

for i=1:num_of_images_coast % Get data for coast class
   current_file_name = coast_data_info(i).name;
   image_path = strcat('../data/image_data/features/coast/',current_file_name);
   image_data = load(image_path);
   data_coast(:,(i-1) * 23 + 1:(i-1) * 23 + 23) = image_data;
end

for i=1:num_of_images_forest % Get data for forest class
   current_file_name = forest_data_info(i).name;
   image_path = strcat('../data/image_data/features/forest/',current_file_name);
   image_data = load(image_path);
   data_forest(:,(i-1) * 23 + 1:(i-1) * 23 + 23) = image_data;
end

%% Get training data for each class. 
train_street_count = uint16(0.7 * num_of_images_street);
train_coast_count = uint16(0.7 * num_of_images_coast);
train_forest_count = uint16(0.7 * num_of_images_forest);

train_data_street = data_street(:,1:train_street_count*23);
train_data_coast = data_coast(:,1:train_coast_count*23);
train_data_forest = data_forest(:,1:train_forest_count*23);

%% Get validation data for each class. 
validation_street_count = uint16(0.15 * num_of_images_street);
validation_coast_count = uint16(0.15 * num_of_images_coast);
validation_forest_count = uint16(0.15 * num_of_images_forest);

validation_data_street = data_street(:,train_street_count*23 + 1: (train_street_count + validation_street_count) * 23);                   
validation_data_coast = data_coast(:,train_coast_count*23 + 1: (train_coast_count + validation_coast_count) * 23);
validation_data_forest = data_forest(:,train_forest_count*23 + 1: (train_forest_count + validation_forest_count) * 23);

%% Get testing data for each class 
test_street_count = validation_street_count;
test_coast_count = validation_coast_count;
test_forest_count = validation_forest_count;

test_data_street = data_street(:,(train_street_count+validation_street_count)*23 + 1:size(data_street,2));                   
test_data_coast = data_coast(:,(train_coast_count+validation_coast_count)*23 + 1:size(data_coast,2));                   
test_data_forest = data_forest(:,(train_forest_count+validation_forest_count)*23 + 1:size(data_forest,2));                   

%% Merge validatin data for all the three classes.
% validation_data = [validation_data_street validation_data_coast validation_data_forest];
% 
% %% Set best value of hyperparameter 'h' such that validation prediction error is minimum. 
% validation_data_count = validation_street_count + validation_coast_count + validation_forest_count;
% validation_score_street = zeros(validation_data_count,1);
% validation_score_coast = zeros(validation_data_count,1);
% validation_score_forest = zeros(validation_data_count,1);
% 
% for i=1:validation_data_count
%     validation_image = validation_data(:,(i-1) * 23 + 1:(i-1) * 23 + 23);
%     validation_score_street(i) = parzen_density_estimate(train_data_street,validation_image,h);
%     validation_score_coast(i) = parzen_density_estimate(train_data_coast,validation_image,h);
%     validation_score_forest(i) = parzen_density_estimate(train_data_forest,validation_image,h);
% end
% 
% %% Plot confusion matrix for validated data.
% actual_validation_class = zeros(3,validation_data_count);
% actual_validation_class(1,1:validation_street_count) = 1;
% actual_validation_class(2,validation_street_count + 1:validation_street_count + validation_coast_count) = 1;
% actual_validation_class(3,validation_street_count + validation_coast_count + 1:validation_data_count) = 1;
% 
% predict_validation_class = zeros(3,validation_data_count);
% 
% for i=1:validation_data_count
%     
%     score_vector = [validation_score_street(i);validation_score_coast(i);validation_score_forest(i)];
%     [M,I] = max(score_vector);
%     predict_validation_class(I,i) = 1;
% end
% 
% plotconfusion(actual_validation_class,predict_validation_class);

%% Check accuracy of bayesian classifier for h=20 i.e hyperparameter with least validation error.

% Combine test data
test_data = [test_data_street test_data_coast test_data_forest];

%% Set best value of hyperparameter 'h' such that validation prediction error is minimum. 
test_data_count = test_street_count + test_coast_count + test_forest_count;
test_score_street = zeros(test_data_count,1);
test_score_coast = zeros(test_data_count,1);
test_score_forest = zeros(test_data_count,1);

for i=1:test_data_count
    testing_image = test_data(:,(i-1) * 23 + 1:(i-1) * 23 + 23);
    test_score_street(i) = parzen_density_estimate(train_data_street,testing_image,h);
    test_score_coast(i) = parzen_density_estimate(train_data_coast,testing_image,h);
    test_score_forest(i) = parzen_density_estimate(train_data_forest,testing_image,h);
end

%% Plot confusion matrix for validated data.
actual_testing_class = zeros(3,test_data_count);
actual_testing_class(1,1:test_street_count) = 1;
actual_testing_class(2,test_street_count + 1:test_street_count + test_coast_count) = 1;
actual_testing_class(3,test_street_count + test_coast_count + 1:test_data_count) = 1;

predict_test_class = zeros(3,test_data_count);

for i=1:test_data_count
    
    score_vector = [test_score_street(i);test_score_coast(i);test_score_forest(i)];
    [M,I] = max(score_vector);
    predict_test_class(I,i) = 1;
end

plotconfusion(actual_testing_class,predict_test_class);















