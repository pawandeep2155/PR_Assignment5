function density_at_x = parzen_density_estimate(training_data,test_point,hyper_parameter)
   data = training_data; % size of data is 36 * (23 * number of images in training)
   x = test_point; % size of x is 36 * 23
   h = hyper_parameter; 
   
   num_of_images = size(data,2)/size(x,2);
   density_row = zeros(size(x,1),1);
   
   for i=1:num_of_images
       
        train_image_i = data(:,(i-1) * size(x,2) + 1:(i-1) * size(x,2) + size(x,2));
        
        for row = 1:size(x,1) % compute density for each row vector of test point.
            density_row(row) = compute_density(train_image_i(row,:)', x(row,:)',h);
        end  
        
   end
   
   density_at_x = mean(density_row);
   
end

function density = compute_density(train_vector, test_vector, hyper_parameter)
   
    h = hyper_parameter;
    distance = test_vector - train_vector;
    distance_norm = distance' * distance;
    density = exp(-distance_norm/(2*h^2));
    
end


















