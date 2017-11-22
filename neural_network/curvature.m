function curve_points = curvature(first_order_data,second_order_data)

    points = zeros(size(second_order_data,1),1);
    for i=1:size(points,1)
        x1 = first_order_data(i+2,1);
        y1 = first_order_data(i+2,2);
        x2 = second_order_data(i,1);
        y2 = second_order_data(i,2);
        points(i,1) = ((x1*y2)-(x2*y1))/((x1^2 + y1^2)^3/2);
    end
    curve_points = points;
end