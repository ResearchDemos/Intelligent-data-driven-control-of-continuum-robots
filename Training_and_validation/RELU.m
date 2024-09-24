function output = RELU(input)

    [row, col] = size(input);
    output = input;
    output(output < 0) = 0;
    % for i = 1:row
    %     for j = 1:col
    % 
    %     end
    % end
end