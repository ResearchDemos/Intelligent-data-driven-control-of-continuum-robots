clear all
close all;

%-------------------------------------------------------------------------%
%----------------- Identify Koopman Model from Data ----------------------%
%-------------------------------------------------------------------------%

train_a_new_model = false;

if  train_a_new_model == true
    % load in data from file
    [ datafile_name , datafile_path ] = uigetfile( 'datafiles/*.mat' , 'Choose data file for sysid...' );
    data4sysid = load( [datafile_path , datafile_name] );
    
    param.lifting_dim = 30;
    dksysid = deep_Koopman( data4sysid, ...
            'snapshots' , Inf ,...                  % Number of snapshot pairs
            'delays' , 1, ...                       % Numer of state/input delays
            'include_input_delay', false, ...       % incporate input delays into the state vector
            'velocity_input_delay', false, ...      % control signal or control velocity (data must has 'ui', the integration of 'u', when True)
            'mini_batch_size', 32, ...              % number of mini_batch_size
            'num_epochs', 60, ...                   % number of epochs
            'step_decay', 0.9, ...                  % decay rate of weight in k steps
            'lifting_dim', param.lifting_dim, ...   % dimension of the lifting space
            'num_encoding_hidden_layer', 128, ...
            'optimizer', 'Adam', ...                % Choose from ['SGDM', 'Adam']
            ...% Specify the options for SGDM optimization
            'learning_decay', 0.01, ...
            'initial_learning_rate', 0.001, ...
            'momentum', 0.9, ...
            'num_steps', 10 ...                     % step number for calculate loss
    );         
    % train linear Koopman model(s)
    dksysid = dksysid.train;
    
    % validate model(s)
    [ results{1} , err{1} ] = dksysid.valNplot_model;

else
    load model/model.mat

    % validate model(s)
    [ results{1} , err{1} ] = dksysid.valNplot_model;
end


