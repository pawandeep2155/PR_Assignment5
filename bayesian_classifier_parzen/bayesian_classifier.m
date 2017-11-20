street_data_info = dir('../data/image_data/features/street/*.jpg_color_edh_entropy');
coast_data_info = dir('../data/image_data/features/coast/*.jpg_color_edh_entropy');
forest_data_info = dir('../data/image_data/features/forest/*.jpg_color_edh_entropy');

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

validation_data_street = data_street(:,train_street_count*23 + 1: train_street_count*23 + 1 + validation_street_count*23);                   
validation_data_coast = data_coast(:,train_coast_count*23 + 1: train_coast_count*23 + 1 + validation_coast_count*23);
validation_data_forest = data_forest(:,train_forest_count*23 + 1: train_forest_count*23 + 1 + validation_forest_count*23);

%% Get testing data for each class 
test_street_count = validation_street_count;
test_coast_count = validation_coast_count;
test_forest_count = validation_forest_count;

% ------------------------ Start work from here ------------------------

%test_data_street = data_street(:,train_street_count*23 + 1: train_street_count*23 + 1 + validation_street_count*23);                   
%test_data_coast = data_coast(:,train_coast_count*23 + 1: train_coast_count*23 + 1 + validation_coast_count*23);
%test_data_forest = data_forest(:,train_forest_count*23 + 1: train_forest_count*23 + 1 + validation_forest_count*23);





