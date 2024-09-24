function ret = get_delay_vector(delay_data)
    
    [row,col] = size(delay_data);
    ret = [];
    for i = 1:col
        ret = [delay_data(:,i); ret];
    end

end