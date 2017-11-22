function first_order_data =  first_derivative(data)
    data_size = size(data,1);
    for i=3:data_size-2
        data(i,1) = (data(i+1,1) - data(i-1,1) + 2*data(i+2,1) - 2*data(i-2,1))/10;
        data(i,2) = (data(i+1,2) - data(i-1,2) + 2*data(i+2,2) - 2*data(i-2,2))/10;
    end
    
    first_order_data = data(3:data_size-2,:);
    
end