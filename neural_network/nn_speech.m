%% Get speech data.
one_data_info = dir('../data/speech_data/isolated/1/*.mfcc');
two_data_info = dir('../data/speech_data/isolated/2/*.mfcc');
z_data_info = dir('../data/speech_data/isolated/z/*.mfcc');

%% Get data for each of the speech digit.
num_of_images_one = length(one_data_info);
num_of_images_two = length(two_data_info);
num_of_images_z = length(z_data_info);

total_features_all_digits = 0;
total_features_one = 0;
total_features_two = 0;
total_features_z = 0;


data_one = [];
data_two = [];
data_z = [];

for i=1:num_of_images_one % Get data for digit one
   current_file_name = one_data_info(i).name;
   image_path = strcat('../data/speech_data/isolated/1/',current_file_name);
   digit_data = dlmread(image_path);
   total_features_all_digits = total_features_all_digits + digit_data(1,2);
   total_features_one = total_features_one + digit_data(1,2);
   digit_data = digit_data(2:size(digit_data,1),:);
   data_one = vertcat(data_one,digit_data);
end

for i=1:num_of_images_two % Get data for digit two
   current_file_name = two_data_info(i).name;
   image_path = strcat('../data/speech_data/isolated/2/',current_file_name);
   digit_data = dlmread(image_path);
   total_features_all_digits = total_features_all_digits + digit_data(1,2);
   total_features_two = total_features_two + digit_data(1,2);
   digit_data = digit_data(2:size(digit_data,1),:);
   data_two = vertcat(data_two,digit_data);
end

for i=1:num_of_images_z % Get data for digit z
   current_file_name = z_data_info(i).name;
   image_path = strcat('../data/speech_data/isolated/z/',current_file_name);
   digit_data = dlmread(image_path);
   total_features_all_digits = total_features_all_digits + digit_data(1,2);
   total_features_z = total_features_z + digit_data(1,2);
   digit_data = digit_data(2:size(digit_data,1),:);
   data_z = vertcat(data_z,digit_data);
end

% Combine all data.
input_data = [data_one ; data_two ; data_z]';

% Target data.
target_data = zeros(3,total_features_all_digits);
target_data(1,1:total_features_one) = 1;
target_data(2,total_features_one + 1:total_features_one + total_features_two) = 1;
target_data(3,total_features_one + total_features_two + 1:size(target_data,2)) = 1;

%% Start neural network
nnstart

