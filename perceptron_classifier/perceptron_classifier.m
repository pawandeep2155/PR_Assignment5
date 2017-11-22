%% Get info for image data. 
street_data_info = dir('../data/image_data/features/street/*.jpg_color_edh_entropy');
coast_data_info = dir('../data/image_data/features/coast/*.jpg_color_edh_entropy');
forest_data_info = dir('../data/image_data/features/forest/*.jpg_color_edh_entropy');

%% Set learning Rate for Gradient Descent.
learn_rate = 0.00001;

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

%% Get testing data for each class 
test_street_count = uint16(0.3 * num_of_images_street);
test_coast_count = uint16(0.3 * num_of_images_coast);
test_forest_count = uint16(0.3 * num_of_images_forest);

test_data_street = data_street(:,train_street_count * 23 + 1:size(data_street,2));                   
test_data_coast = data_coast(:,train_coast_count * 23 + 1:size(data_coast,2));                   
test_data_forest = data_forest(:,train_forest_count * 23 + 1:size(data_forest,2));                   

%% Take pair 1-2,3 and compute plane equation.
training_data = [train_data_street train_data_coast train_data_forest];
training_points_class123 = train_street_count + train_coast_count + train_forest_count;
fprintf('Training on 1-2,3 \n');
w1 = randi(10,24,1);

for i=1:100
  
    for j=1:training_points_class123
        
        training_image = training_data(:,(j-1) * 23 + 1:(j-1) * 23 + 23);

        % For each row of training each compute w'x.
        for k=1:size(data_street,1)
            if(j <= train_street_count && w1' * [training_image(k,:)' ; 1] <= 0)
                w1 = w1 + learn_rate * [training_image(k,:)' ; 1];
            end
            if(j > train_street_count &&  w1' * [training_image(k,:)' ; 1] >= 0)
                w1 = w1 - learn_rate * [training_image(k,:)' ; 1];
            end    
        end

    end
end

%% Take pair 2-3,1 and compute plane equation.
training_data = [train_data_coast train_data_forest train_data_street];
training_points_class231 = train_coast_count + train_forest_count + train_street_count;
fprintf('Training on 2-1,3 \n');
w2 = randi(10,24,1);

for i=1:100
  
    for j=1:training_points_class231
        
        training_image = training_data(:,(j-1) * 23 + 1:(j-1) * 23 + 23);
        
        % For each row of training each compute w'x.
        for k=1:size(data_coast,1)
            if(j <= train_coast_count && w2' * [training_image(k,:)' ; 1] <= 0)
                w2 = w2 + learn_rate * [training_image(k,:)' ; 1];
            end
            if(j > train_coast_count && w2' * [training_image(k,:)' ; 1] >= 0)
                w2 = w2 - learn_rate * [training_image(k,:)' ; 1];
            end    
        end
    end
end

%% Take pair 3-1,2 and compute plane equation.
training_data = [train_data_forest train_data_street train_data_coast];
training_points_class312 = train_forest_count + train_street_count + train_coast_count;
fprintf('Training on 3-1,2 \n');
w31 = randi(10,24,1);

for i=1:100
  
    for j=1:training_points_class312
        
        training_image = training_data(:,(j-1) * 23 + 1:(j-1) * 23 + 23);
        
        % For each row of training each compute w'x.
        for k=1:size(data_forest,1)
            if(j <= train_forest_count && w31' * [training_image(k,:)' ; 1] <= 0)
                w31 = w31 + learn_rate * [training_image(k,:)' ; 1];
            end
            if(j > train_forest_count && w31' * [training_image(k,:)' ; 1] >= 0)
                w31 = w31 - learn_rate * [training_image(k,:)' ; 1];
            end    
        end
    end
end

%% Test in which class test data belongs.
w = [w1 w2 w31];
test_data = [test_data_street test_data_coast test_data_forest ];
test_points = test_street_count + test_coast_count + test_forest_count;
fprintf('Testing...\n');
predicted_class = zeros(3,test_points);

for i=1:test_points
    
    testing_image = test_data(:,(i-1) * 23 + 1:(i-1) * 23 + 23);
    test_class_1 = 0;
    test_class_2 = 0;    
    test_class_3 = 0;
    
    for j=1:3
        
        for k=1:size(data_street,1)
            if(w(:,j)' * [testing_image(k,:)' ; 1] >= 0)
                 if(j == 1)
                    test_class_1 = test_class_1 + 1;
                 elseif (j==2)
                     test_class_2 = test_class_2 + 1;
                 else 
                     test_class_3 = test_class_3 + 1;
                 end
            end
        end
                
    end
    
    test_class = [test_class_1 ; test_class_2 ; test_class_3];
    [M,I] = max(test_class);
    predicted_class(I,i) = 1;
    
end

%% Draw confusion matrix.
actual_class = zeros(3,test_points);
actual_class(1,1:test_street_count) = 1;
actual_class(2,test_street_count + 1:test_street_count + test_coast_count) = 1;
actual_class(3,test_street_count + test_coast_count + 1:test_points) = 1;

plotconfusion(actual_class,predicted_class);












