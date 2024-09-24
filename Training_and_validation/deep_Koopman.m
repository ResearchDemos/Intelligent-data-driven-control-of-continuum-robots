classdef deep_Koopman
    %deep_Koopman: deep Koopman net class
    %   Detailed explanation goes here
    
    properties
        params; % paramaters of the system
        model;  % linear model of the system
        lift;   % lifting functions for system
        basis;  % symbolic basis set of observables
        
        net; % deep neural network
        subnet; % state transition network (without encoding/lifting net)
 
        scaledown;  % functions for scaling to [-1,1]
        scaleup;    % functions for scaling from [-1,1]
          

        traindata;  % scaled exp/sim data for training the model
        valdata;    % scaled exp/sim data for validating the model
        snapshotPairs; % snapshot pairs extracted from training data
    end
    
    methods
        % CLASS CONSTRUCTOR
        function obj = deep_Koopman( data4sysid, varargin )
            %kmpc: Construct an instance of this class
            %   sysid_class - sysid class object with a model and params
            %   sysid_class_s - sysid class object with static snapshotPairs
            %    properties
            %   varargin - Name, Value pairs for class properties
            
            % take some properties/methods from the sysid class
%             obj.params = sysid_class.params;
%             obj.model = sysid_class.model;
%             obj.lift = sysid_class.lift;
%             obj.basis = sysid_class.basis;
%             obj.get_zeta = @sysid_class.get_zeta;   % copies this method for convenience
%             obj.scaledown = sysid_class.scaledown;
%             obj.scaleup = sysid_class.scaleup;
            
            
            % process data
            if isfield( data4sysid , 'train_lqr' ) || isfield( data4sysid , 'val_lqr' )
                data4sysid.train = data4sysid.train_lqr;
                data4sysid.val = data4sysid.val_lqr;
            end
            % verify that data4sysid has required fields
            if ~isfield( data4sysid , 'train' ) || ~isfield( data4sysid , 'val' )
                error('Input must have *train* and *val* fields of type cell array');
            end
            % isolate one trial to extract some model parameters
            data = data4sysid.train{1};
            data4train = data4sysid.train;
            data4val = data4sysid.val;
            
            % set param values based on the data
            obj.params = struct;    % initialize params struct
            obj.params.n = size( data.y , 2 );  % dimension of measured state
            obj.params.m = size( data.u , 2 );  % dimension of input
            obj.params.Ts = mean( data.t(2:end) - data.t(1:end-1) );    % sampling time
            
            % if data has a params field save it as sysParams
            if isfield( data , 'params' )
                obj.params.sysParams = data.params;
                obj.params.isfake = true;   % if params field exist the system is fake
            end
            
            % set defualt values of Name, Value optional arguments
            obj.params.num_snapshots = Inf; % number of snapshot pairs to use in training
            obj.params.delays = 1; % number of delays to include, default 1 to ensure model is dynamic
            obj.params.num_steps = 1;  % number of steps for calculating loss
            obj.params.include_input_delay=false; % true or false, denoting whether incorporating input delay in state
            obj.params.velocity_input_delay=false; % true or false, denoting incorating input delay or velocity-level input delay
            % Specify the options for optimization.
            obj.params.velocity = [];
            obj.params.averageGrad = [];
            obj.params.averageSqGrad = [];

            % replace default values with user input values
            obj = obj.parse_args( varargin{:} );
            
            obj.params.nd = obj.params.delays;  % saves copy in params (for consistency with mpc class)
            if obj.params.include_input_delay
                obj.params.nzeta = obj.params.n * ( obj.params.delays + 1 ) + obj.params.m * obj.params.delays;
            else 
                obj.params.nzeta = obj.params.n * ( obj.params.delays + 1 );
            end

            % merge the training data into a single big file (requred for training function to work)
            data4train_merged = obj.merge_trials( data4train );
            
            % scale data to be in range [-1 , 1]
            [ traindata , obj ] = obj.get_scale( data4train_merged );
            valdata = cell( size( data4val ) );
            for i = 1 : length( data4val )
                valdata{i} = obj.scale_data( data4val{i} );
            end
            obj.traindata = traindata;
            obj.valdata = valdata;
            
            % get shapshot pairs from traindata
            obj.snapshotPairs = obj.get_snapshotPairs( obj.traindata , obj.params.num_snapshots );
            
            % Initialize deep Koopman neural network
            obj.net = dlnetwork; 
            obj.subnet =  dlnetwork; 
            tempNet = [
                sequenceInputLayer(obj.params.m,"Name","u_k")
                fullyConnectedLayer(obj.params.lifting_dim,"Name","Bu_k")];
            obj.net = addLayers(obj.net,tempNet);
            obj.subnet = addLayers(obj.subnet,tempNet);

            tempNet = sequenceInputLayer(obj.params.nzeta,"Name","x_k");
            obj.net = addLayers(obj.net,tempNet);
            tempNet = sequenceInputLayer(obj.params.lifting_dim,"Name","z");
            obj.subnet = addLayers(obj.subnet,tempNet);

            tempNet = [
                fullyConnectedLayer(obj.params.num_encoding_hidden_layer,"Name","fc")
                fullyConnectedLayer(obj.params.lifting_dim-obj.params.nzeta,"Name","phix_k")
                reluLayer("Name","relu")];
            obj.net = addLayers(obj.net,tempNet);

            tempNet = [
                concatenationLayer(1,2,"Name","z")
                fullyConnectedLayer(obj.params.lifting_dim,"Name","Az")];
            obj.net = addLayers(obj.net,tempNet);
            tempNet = fullyConnectedLayer(obj.params.lifting_dim,"Name","Az");
            obj.subnet = addLayers(obj.subnet,tempNet);

            tempNet = additionLayer(2,"Name","dz");
            obj.net = addLayers(obj.net,tempNet);
            obj.subnet = addLayers(obj.subnet,tempNet);

            % Clear auxiliary variables
            clear tempNet;

            obj.net = connectLayers(obj.net,"Bu_k","dz/in1");
            obj.net = connectLayers(obj.net,"x_k","fc");
            obj.net = connectLayers(obj.net,"x_k","z/in1");
            obj.net = connectLayers(obj.net,"relu","z/in2");
            obj.net = connectLayers(obj.net,"Az","dz/in2");
            obj.net = initialize(obj.net);

            obj.subnet = connectLayers(obj.subnet,"Bu_k","dz/in1");
            obj.subnet = connectLayers(obj.subnet,"z","Az");
            obj.subnet = connectLayers(obj.subnet,"Az","dz/in2");
            obj.subnet = initialize(obj.subnet);
        end
        
        % parse_args: Parses the Name, Value pairs in varargin
        function obj = parse_args( obj , varargin )
            %parse_args: Parses the Name, Value pairs in varargin of the
            % constructor, and assigns property values
            for idx = 1:2:length(varargin)
                obj.params.(varargin{idx}) = varargin{idx+1} ;
            end
        end
            %% operations on simulation/experimental data (some are redundant and found in the data class)
        
        function [ data_scaled , obj ] = get_scale( obj , data )
            %scale: Scale sim/exp data to be in range [-1 , 1]
            %    Also creates scaleup/scaledown matrices and saves as params
            %    data - struct containing fields t , y , u (at least)
            %    data_scaled - struct containing t , y , u , x (optional)   
            
            % get min/max values in each dimension
            y_min = min( data.y );
            u_min = min( data.u );
            y_max = max( data.y );
            u_max = max( data.u );
            dy_min = min( data.dy );
            du_min = min( data.du );
            dy_max = max( data.dy );
            du_max = max( data.du );
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                ui_min = min( data.ui);
                ui_max = max( data.ui);
            end
            
            % calculate centers of range
            y_dc = ( y_max + y_min ) ./ 2;
            u_dc = ( u_max + u_min ) ./ 2;
            dy_dc = ( dy_max + dy_min ) ./ 2;
            du_dc = ( du_max + du_min ) ./ 2;
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
            	ui_dc = (ui_max + ui_min) ./ 2;
            end
            
            % calculate scaling factors
            scale_y = ( y_max - y_min ) ./ 2;
            scale_u = ( u_max - u_min ) ./ 2;
            scale_dy = ( dy_max - dy_min ) ./ 2;
            scale_du = ( du_max - du_min ) ./ 2;
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                scale_ui = (ui_max - ui_min) ./ 2;
            end
            
            % shift and scale the data
            data_scaled = struct;    % initialize
            data_scaled.t = data.t;  % time is not scaled
            data_scaled.y = ( data.y - y_dc ) ./ scale_y;
            data_scaled.u = ( data.u - u_dc ) ./ scale_u;
            data_scaled.dy = ( data.dy - dy_dc ) ./ scale_dy;
            data_scaled.du = ( data.du - du_dc ) ./ scale_du;
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                data_scaled.ui = (data.ui - ui_dc) ./ scale_ui;
            end
            
            % save scaling functions
            y = sym( 'y' , [ 1 , obj.params.n ] );
            u = sym( 'u' , [ 1 , obj.params.m ] );
            dy = sym( 'dy' , [ 1 , obj.params.n ] );
            du = sym( 'du' , [ 1 , obj.params.m ] );
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                ui = sym('ui' , [1, obj.params.m] );
            end
            y_scaledown = ( y - y_dc ) ./ scale_y;
            u_scaledown = ( u - u_dc ) ./ scale_u;
            dy_scaledown = ( dy - dy_dc ) ./ scale_dy;
            du_scaledown = ( du - du_dc ) ./ scale_du;
            obj.scaledown.y = matlabFunction( y_scaledown , 'Vars' , {y} );
            obj.scaledown.u = matlabFunction( u_scaledown , 'Vars' , {u} );
            obj.scaledown.dy = matlabFunction( dy_scaledown , 'Vars' , {dy} );
            obj.scaledown.du = matlabFunction( du_scaledown , 'Vars' , {du} );
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                ui_scaledown = ( ui - ui_dc ) ./ scale_ui;
                obj.scaledown.ui = matlabFunction( ui_scaledown , 'Vars' , {ui} );
            end
            
            y_scaleup = ( y .* scale_y ) + y_dc;
            u_scaleup = ( u .* scale_u ) + u_dc;
            dy_scaleup = ( dy .* scale_dy ) + dy_dc;
            du_scaleup = ( du .* scale_du ) + du_dc;
            obj.scaleup.y = matlabFunction( y_scaleup , 'Vars' , {y} );
            obj.scaleup.u = matlabFunction( u_scaleup , 'Vars' , {u} );
            obj.scaleup.dy = matlabFunction( dy_scaleup , 'Vars' , {dy} );
            obj.scaleup.du = matlabFunction( du_scaleup , 'Vars' , {du} );
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
            	ui_scaleup = ( ui .* scale_ui ) + ui_dc;
                obj.scaleup.ui = matlabFunction( ui_scaleup , 'Vars' , {ui} );
            end
            
            % save scaling factors
            obj.params.scale.y_factor = scale_y;
            obj.params.scale.y_offset = y_dc;
            obj.params.scale.u_factor = scale_u;
            obj.params.scale.u_offset = u_dc;
            obj.params.scale.dy_factor = scale_dy;
            obj.params.scale.dy_offset = dy_dc;
            obj.params.scale.du_factor = scale_du;
            obj.params.scale.du_offset = du_dc;
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                obj.params.scale.ui_factor = scale_ui;
                obj.params.scale_ui_offset = ui_dc;
            end
            
            % do same for x if it is part of data struct
            if ismember( 'x' , fields(data) )
                x_min = min( data.x );
                x_max = max( data.x );
                x_dc = ( x_max + x_min ) ./ 2;
                scale_x = ( x_max - x_min ) ./ 2;
                data_scaled.x = ( data.x - x_dc ) ./ scale_x;
                x = sym( 'x' , [ 1 , size(data.x,2) ] );
                x_scaledown = ( x - x_dc ) ./ scale_x;
                obj.scaledown.x = matlabFunction( x_scaledown , 'Vars' , {x} );
                x_scaleup = ( x .* scale_x ) + x_dc;
                obj.scaleup.x = matlabFunction( x_scaleup , 'Vars' , {x} );
            end
            
            % create scaling functions for zeta
            zeta = sym( 'zeta' , [ 1 , obj.params.nzeta ] );
            zeta_scaledown = sym( zeros( size(zeta) ) );
            zeta_scaleup = sym( zeros( size(zeta) ) );
            zeta_scaledown(1:obj.params.n) = obj.scaledown.y( zeta(1:obj.params.n) );
            zeta_scaleup(1:obj.params.n) = obj.scaleup.y( zeta(1:obj.params.n) );
            
            dzeta = sym( 'dzeta' , [ 1 , obj.params.nzeta ] );
            dzeta_scaledown = sym( zeros( size(zeta) ) );
            dzeta_scaleup = sym( zeros( size(zeta) ) );
            dzeta_scaledown(1:obj.params.n) = obj.scaledown.dy( dzeta(1:obj.params.n) );
            dzeta_scaleup(1:obj.params.n) = obj.scaleup.dy( dzeta(1:obj.params.n) );
            
            for i = 1 : obj.params.delays   % for y delays
                range = obj.params.n * i + 1 : obj.params.n * (i+1);
                zeta_scaledown(range) = obj.scaledown.y( zeta(range) );
                zeta_scaleup(range) = obj.scaleup.y( zeta(range) );
                
                dzeta_scaledown(range) = obj.scaledown.dy( dzeta(range) );
                dzeta_scaleup(range) = obj.scaleup.dy( dzeta(range) );
            end
            if obj.params.include_input_delay
                if obj.params.velocity_input_delay
                    for i = 1 : obj.params.delays   % for u delays
                        endy = obj.params.n * ( obj.params.delays + 1 );
                        range = endy + obj.params.m * (i-1) + 1 : endy + obj.params.m * i;
                        zeta_scaledown(range) = obj.scaledown.u( zeta(range) );
                        zeta_scaleup(range) = obj.scaleup.u( zeta(range) );

                        dzeta_scaledown(range) = obj.scaledown.du( dzeta(range) );
                        dzeta_scaleup(range) = obj.scaleup.du( dzeta(range) );
                    end
                else
                    for i = 1 : obj.params.delays   % for u delays
                        endy = obj.params.n * ( obj.params.delays + 1 );
                        range = endy + obj.params.m * (i-1) + 1 : endy + obj.params.m * i;
                        zeta_scaledown(range) = obj.scaledown.ui( zeta(range) );
                        zeta_scaleup(range) = obj.scaleup.ui( zeta(range) );
                        
                        dzeta_scaledown(range) = obj.scaledown.u( dzeta(range) );
                        dzeta_scaleup(range) = obj.scaleup.u( dzeta(range) );
                    end
                end
            end
            obj.scaledown.zeta = matlabFunction( zeta_scaledown , 'Vars' , {zeta} );
            obj.scaleup.zeta = matlabFunction( zeta_scaleup , 'Vars' , {zeta} );
            obj.scaledown.dzeta = matlabFunction( dzeta_scaledown , 'Vars' , {dzeta} );
            obj.scaleup.dzeta = matlabFunction( dzeta_scaleup , 'Vars' , {dzeta} );
        end
        
        % resample (resamples data with a desired time step)
        function data_resampled = resample( obj , data , Ts )
            %resample: resamples sim/exp data with a desired timestep
            %   data - struct with fields t, y, x (optional)
            %   Ts - the desired sampling period
            
            % get query points
            tq = ( data.t(1) : Ts : data.t(end) )';
            
            data_resampled.t = tq;
            data_resampled.u = interp1( data.t , data.u , tq );
            data_resampled.y = interp1( data.t , data.y , tq );
            data_resampled.du = interp1( data.t , data.du , tq );
            data_resampled.dy = interp1( data.t , data.dy , tq );
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                data_resampled.ui = interp1( data.t , data.ui , tq );
            end
            if ismember( 'x' , fields(data) )
                data_resampled.x = interp1( data.t , data.x , tq );
            end
        end
        
        % scale_data (scale sim/exp data to be in range [-1 , 1])
        function data_scaled = scale_data( obj , data , down )
            %scale: Scale sim/exp data based on the scalefactors set in
            % get_scale.
            %    data - struct containing fields t , y , u (at least)
            %    data_scaled - struct containing t , y , u , x (optional)
            %    down - boolean. true to scale down, false to scale up.
            
            if nargin < 3
                down = true; % default is to scale down
            end
            
            % scale the data
            if down
                data_scaled = struct;    % initialize
                data_scaled.t = data.t;  % time is not scaled
                data_scaled.y = obj.scaledown.y( data.y );  %data.y * obj.params.scaledown.y;
                data_scaled.u = obj.scaledown.u( data.u );  %data.u * obj.params.scaledown.u;
                data_scaled.dy = obj.scaledown.dy( data.dy );  %data.dy * obj.params.scaledown.dy;
                data_scaled.du = obj.scaledown.du( data.du );  %data.du * obj.params.scaledown.du;
                if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                    data_scaled.ui = obj.scaledown.ui( data.ui);
                end
                if ismember( 'x' , fields(data) )
                    data_scaled.x = obj.scaledown.x( data.x );  %data.x * obj.params.scaledown.x;
                end
            else
                data_scaled = struct;    % initialize
                data_scaled.t = data.t;  % time is not scaled
                data_scaled.y = obj.scaleup.y( data.y );    %data.y * obj.params.scaleup.y;
                data_scaled.u = obj.scaleup.u( data.u );    %data.u * obj.params.scaleup.u;
                data_scaled.dy = obj.scaleup.dy( data.dy );    %data.y * obj.params.scaleup.y;
                data_scaled.du = obj.scaleup.du( data.du );    %data.u * obj.params.scaleup.u;
                if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                    data_scaled.ui = obj.scaleup.ui( data.ui ); 
                end
                if ismember( 'x' , fields(data) )
                    data_scaled.x = data.scaleup.x( data.x );    %data.x * obj.params.scaleup.x;
                end
            end
        end
        
        % chop (chop data into several trials)
        function data_chopped = chop( obj , data , num , len )
            %chop: chop data into num trials of lenght len
            %   data - struct with fields t , y , (x)
            %   data_chopped - cell array containing the chopped datas
            
            % determine length of timestep
            Ts = mean( data.t(2:end) - data.t(1:end-1) ); % take mean in case they're not quite uniform
            
            % find maximum length of each chop for given num
            maxlen = data.t(end) / num;
            if len > maxlen
                len = maxlen;
                disp([ 'Maximum trial length is ' , num2str(maxlen) , 's. Using this value instead.' ]);
            end
            
            % set length of the chops in terms of time steps
            lenk = length( find( data.t < len ) );
            maxlenk = length( find( data.t < maxlen ) );
            
            data_chopped = cell( 1 , num );
            for i = 1 : num
                index = (i-1) * maxlenk + ( 1 : lenk );
                
                % chop the data
                data_chopped{i}.t = ( ( 1 : lenk ) - 1 ) * Ts;
                data_chopped{i}.y = data.y( index , : );
                data_chopped{i}.u = data.u( index , : );
                data_chopped{i}.dy = data.dy( index , : );
                data_chopped{i}.du = data.du( index , : );
                if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                    data_chopped{i}.ui = data.ui( index , : );
                end
                if ismember( 'x' , fields(data) )
                    data_chopped{i}.x = data.x( index , : );
                end
            end  
        end
        
        
        function data_disentangled = disentangle_data(obj, data)
%             t = data.t;
%             u_t = data.u;
            S_t = data.y;

            S_f = fft(S_t);
            L = size(S_f,1);

            S_f_abs = abs(S_f);
            S_f_abs_sorted = sort(S_f_abs);
            invariant_percentage = 0.5;
            critical_value = S_f_abs_sorted(round(L*(1-invariant_percentage)),:);
            invariant_mask = zeros(L,size(S_f,2));
            variant_mask = zeros(L,size(S_f,2));
            for i = 1:size(S_t,2)
               invariant_mask((S_f_abs(:,i) >= critical_value(i)),i) = 1; 
               variant_mask((S_f_abs(:,i) < critical_value(i)),i) = 1; 
            end
            S_f_invariant = S_f .* invariant_mask;
            S_f_variant = S_f .* variant_mask;
            S_t_invariant = ifft(S_f_invariant);
            S_t_variant = ifft(S_f_variant);
            
            data_disentangled = data;
%             data_disentangled.y=S_t_invariant;
        end
        
        % merge_trials (merge cell array containing several sim/exp trials into one data struct)
        function data_merged = merge_trials( obj , data )
            %merge_trials: Merge cell array containing several sim/exp trials into one data struct
            %   data - cell array where each cell is a data struct with
            %   fields t, y, u, (x), (params), ...
            %   data_merged: data struct with the same fields
            
            % confirm that data is a cell array (i.e. contains several trials)
            % If so, concatenate all trials into a single data struct 
            if iscell( data )
                data_merged = obj.disentangle_data(data{1});  % initialize the merged data struct
                for i = 2 : length( data )
                    data_fields = fields( data{i} );
                    data_i = obj.disentangle_data(data{i});
                    for j = 1 : length( data_fields )
                        if isa( data_i.( data_fields{j} ) , 'numeric' )
                            data_merged.( data_fields{j} ) = [ data_merged.( data_fields{j} ) ; data_i.( data_fields{j} ) ];
                        end
                    end
                end
            else
                data_merged = data; % if not a cell array, do nothing
            end
        end
            
        %% save the class object
        
        % save_class
        function obj = save_class( obj  )
            %save_class: Saves the class as a .mat file
            %   If class is from a simulated system, it is saved in the
            %   subfolder corresponding to that system.
            %   If class if from a real system, it is saved in the generic
            %   folder /systems/fromData/.
            %   varargin = isupdate - indicates whether this is an update of a
            %   previously saved class (1) or a new class (0).
            
            dateString = datestr(now , 'yyyy-mm-dd_HH-MM'); % current date/time
            classname = [ 'n-' , num2str( obj.params.n ) , '_m-' , num2str( obj.params.m ) , '_del-' , num2str( obj.params.nd ) , '_' , dateString ];
            obj.params.classname = classname;   % create classname parameter

            % save the class file
            dirname = [ 'systems' , filesep , 'fromData' ];
            fname = [ dirname , filesep , classname, '.mat' ];
            save( fname , 'sysid_class' );

        end
        
%%    
        % get_zeta (adds a zeta field to a test data struct)
        function [ data_out , zeta, dzeta ] = get_zeta( obj , data_in )
            %get_zeta: Adds a zeta field to a test data struct
            %   data_in - struct with t , x , y , u fields
            %   zeta - [ y , yd1 , yd2 , ... , ud1 , ud2 , ... ]
            
            data_out = data_in;
            
            % add the zeta field
            for i = obj.params.nd + 1 : size( data_in.y , 1 )
                ind = i - obj.params.nd;    % current timestep index
                y = data_in.y( i , : );
                u = data_in.u( i , : );
                dy = data_in.dy( i , : );
                du = data_in.du( i , : );
                if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                    ui = data_in.ui( i , : );
                end
                
                
                ydel = zeros( 1 , obj.params.nd * obj.params.n );
                udel = zeros( 1 , obj.params.nd * obj.params.m );
                dydel = zeros( 1 , obj.params.nd * obj.params.n );
                dudel = zeros( 1 , obj.params.nd * obj.params.m );
                if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                    uidel = zeros( 1 , obj.params.nd * obj.params.m );
                end
                for j = 1 : obj.params.nd
                    fillrange_y = obj.params.n * (j - 1) + 1 : obj.params.n * j;
                    fillrange_u = obj.params.m * (j - 1) + 1 : obj.params.m * j;
                    ydel(1 , fillrange_y) = data_in.y( i - j , : );
                    udel(1 , fillrange_u) = data_in.u( i - j , : );
                    dydel(1 , fillrange_y) = data_in.dy( i - j , : );
                    dudel(1 , fillrange_u) = data_in.du( i - j , : );
                    if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                        uidel(1 , fillrange_u) = data_in.ui( i - j , : );
                    end
                end
                if obj.params.include_input_delay
                    if obj.params.velocity_input_delay
                        zetak = [ y , ydel , udel ];
                        dzetak = [dy, dydel, dudel];
                    else
                        zetak = [ y , ydel , uidel ];
                        dzetak = [dy, dydel, udel];
                    end
                else
                    zetak = [ y , ydel  ];
                    dzetak = [dy, dydel ];
                end

%                 if obj.liftinput == 1     % include input in zeta
%                     zetak = [ zetak , u ];
%                 end
                data_out.zeta( ind , : ) = zetak;
                data_out.dzeta( ind , : ) = dzetak;
                data_out.uzeta( ind , : ) = data_in.u( i , : );    % current timestep with zeta (input starting at current timestep)
                data_out.duzeta( ind , : ) = data_in.du( i , : );  
                if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                    data_out.uizeta( ind , : ) = data_in.ui( i , : );
                end
            end
            zeta = data_out.zeta;
            dzeta = data_out.dzeta;
        end
        
        % get_snapshotPairs (convert time-series data into snapshot pairs)
        function snapshotPairs = get_snapshotPairs( obj , data , varargin )
            %get_snapshotPairs: Convert time-series data into a set of num
            %snapshot pairs.
            %   data - struct with fields x , y , u , t , (zeta) OR cell
            %     array containing cells which contain those fields
            %   varargin = num - number of snapshot pairs to be taken
            
            % check wheter data is a cell array (i.e. contains several trials)
            % If so, concatenate all trials into a single data struct 
            if iscell( data )
                data_merged = obj.merge_trials( data );
                data = data_merged; % replace cell array with merged data struct
            end
            
            % check if data has a zeta field, create one if not
            if ~ismember( 'zeta' , fields(data) )
                data = obj.get_zeta( data );
            end
            
            % separate data into 'before' and 'after' time step
            before.t = data.t( obj.params.nd + 1 : end-1 );
            before.zeta = data.zeta( 1:end-1 , : );
            before.dzeta = data.dzeta( 1:end-1, : );

            after.t = data.t( obj.params.nd + 2 : end );
            after.zeta = data.zeta( 2:end , : );
            after.dzeta = data.dzeta( 1:end-1 , : );
            u = data.uzeta( 1:end-1 , : );    % input that happens between before.zeta and after.zeta
            du = data.duzeta( 1:end-1 , : );
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                ui = data.uizeta( 1:end-1 , : );
            end

            data_amount = length(before.t);
            % prepare k-step data
            % X_i = zeros(obj.params.nzeta,obj.params.num_steps);
            % U_i = zeros(obj.params.m,obj.params.num_steps);
            % T_i = zeros(obj.params.nzeta,obj.params.num_steps);

            X = zeros(obj.params.nzeta,obj.params.num_steps,1);
            dX = zeros(obj.params.nzeta,obj.params.num_steps,1);
            U = zeros(obj.params.m,obj.params.num_steps,1);
            % dU = zeros(obj.params.m,obj.params.num_steps,1);
            % UI = zeros(obj.params.m,obj.params.num_steps,1);
            T = zeros(obj.params.nzeta,obj.params.num_steps,1);
            dT = zeros(obj.params.nzeta,obj.params.num_steps,1);
            pair_num=0;
            for i = 1:data_amount-obj.params.num_steps
                if before.t(i+obj.params.num_steps-1) >= before.t(i) && after.t(i+obj.params.num_steps-1) >= after.t(i)
                   pair_num = pair_num + 1;
                   X(:,:,pair_num) = before.zeta(i:i+obj.params.num_steps-1,:)';
                   dX(:,:,pair_num) = before.dzeta(i:i+obj.params.num_steps-1,:)';
                   U(:,:,pair_num) = u(i:i+obj.params.num_steps-1,:)';
                   % dU(:,:,pair_num) = du(i:i+obj.params.num_steps-1,:)';
                   T(:,:,pair_num) = after.zeta(i:i+obj.params.num_steps-1,:)';
                   dT(:,:,pair_num) = after.dzeta(i:i+obj.params.num_steps-1,:)';
                   % if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                   %     UI(:,:,pair_num) = ui(i:i+obj.params.num_steps-1,:)';
                   % end
                end
            end

            
            % % remove pairs that fall at the boundary between sim/exp trials
            % goodpts = find( before.t < after.t );
            % before.zeta = before.zeta( goodpts , : );
            % before.dzeta = before.dzeta( goodpts , : );
            % after.zeta = after.zeta( goodpts , : );
            % after.dzeta = after.dzeta( goodpts , : );
            % u = u( goodpts , : );
            % du = du( goodpts , : );
            % if obj.params.include_input_delay && ~obj.params.velocity_input_delay
            %     ui = ui( goodpts , : );
            % end
            % % set the number of snapshot pairs to be taken
            % num_max = size( before.zeta , 1 ) - 1; % maximum number of snapshot pairs
            % if length(varargin) == 1
            %     num = varargin{1};
            %     if num > num_max - 1
            %         message = [ 'Number of snapshot pairs cannot exceed ' , num2str(num_max) , '. Taking ' , num2str(num_max) , ' pairs instead.' ];
            %         disp(message);
            %         num = num_max;
            %     end
            % else
            %     num = num_max;
            % end
            % 
            % % randomly select num snapshot pairs
            % total = num_max;
            % s = RandStream('mlfg6331_64'); 
            % index = datasample(s , 1:total, num , 'Replace' , false);
            % 
            % snapshotPairs.alpha = before.zeta( index , : ); 
            % snapshotPairs.beta = after.dzeta( index , : );
            % snapshotPairs.u = u( index , : );
            % snapshotPairs.du = du( index , : );
            % if obj.params.include_input_delay && ~obj.params.velocity_input_delay
            %     snapshotPairs.ui = ui( index , : );
            % end

            snapshotPairs.X = X;
            snapshotPairs.dX = dX;
            snapshotPairs.U = U;
            snapshotPairs.T = T;
            snapshotPairs.dT = dT;
        end

        %% train the network

        function [obj] = train(obj)

            [~,~,num_snapshots] =size(obj.snapshotPairs.X);
            obj.params.num_snapshots = num_snapshots; % number of total data pairs
            X = obj.snapshotPairs.X;
            T = obj.snapshotPairs.T;
            dT = obj.snapshotPairs.dT;
            U = obj.snapshotPairs.U;
            % trasnform data to datastore
            dsX = arrayDatastore(X,IterationDimension=3);
            dsT = arrayDatastore(T,IterationDimension=3);
            dsdT = arrayDatastore(dT,IterationDimension=3);
            dsU = arrayDatastore(U,IterationDimension=3);
            dsTrain = combine(dsX,dsU,dsT,dsdT);
            % Mini-batch queue
            mbq = minibatchqueue(dsTrain,...
                MiniBatchSize=obj.params.mini_batch_size);

            num_snapshots = obj.params.num_snapshots;
            mini_batch_size = obj.params.mini_batch_size;
            num_epochs = obj.params.num_epochs;
            optimizer = obj.params.optimizer;
            initial_learning_rate = obj.params.initial_learning_rate;
            learning_decay = obj.params.learning_decay;
            momentum = obj.params.momentum;
            velocity = obj.params.velocity;
            averageGrad = obj.params.averageGrad;
            averageSqGrad = obj.params.averageSqGrad;
            Ts = obj.params.Ts;
            step_decay = obj.params.step_decay;


            numObservationsTrain = num_snapshots;
            numIterationsPerEpoch = ceil(numObservationsTrain / mini_batch_size);
            numIterations = num_epochs * numIterationsPerEpoch;
            monitor = trainingProgressMonitor( ...
                Metrics="Loss", ...
                Info=["Epoch" "LearnRate"], ...
                XLabel="Iteration");

            epoch = 0;
            iteration = 0;

            % Loop over epochs
            while epoch < num_epochs && ~monitor.Stop
                epoch = epoch + 1;

                shuffle(mbq);

                % Loop ove mini-batches
                while hasdata(mbq) && ~monitor.Stop
                    iteration = iteration + 1;

                    % Read mini-batch of data
                    [X,U,T,dT] = next(mbq);

                    % Evaluate the model gradients, state, and loss using dlfeval and the
                    % modelLoss function and update the network state.
                    [loss,gradients,state] = dlfeval(@obj.model_loss, obj.net, obj.subnet, X, U, T, dT, Ts, step_decay);
                    obj.net.State = state;


                    switch optimizer
                        case 'SGDM'
                            % Determine learning rate for time-based decay learning rate schedule.
                            learnRate = initial_learning_rate/(1 + learning_decay*iteration);
                            % Update the network parameters using the SGDM optimizer.
                            [obj.net,velocity] = sgdmupdate(obj.net,gradients,velocity,learnRate,momentum);
                        case 'Adam'
                            % Update the network parameters using the Adam optimizer.
                            [obj.net,averageGrad,averageSqGrad] = adamupdate(obj.net,gradients,averageGrad,averageSqGrad,iteration);
                            learnRate = 0.001; % default
                        otherwise
                            disp('Please choose a valid optimizer from ["SGDM","Adam"]');
                    end

                    % Update the training progress monitor.
                    recordMetrics(monitor,iteration,Loss=loss);
                    updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
                    monitor.Progress = 100 * iteration/(numIterations);

                end


            end

        end
                
        function [loss,gradients,state] = model_loss(obj, net, subnet, X, U, T, dT, Ts, decay_rate)
            %  X U dT contain a batch of sequential data within k time intervals
            % The data is time-ordered in the step/second dimension, but not in the
            % batch dimension
            % By equence data, we mean X(:, i, j) corresponds time instant t_i, where
            % t_i is the start time of the sequence data

            % Ts = obj.params.Ts;
            % decay_rate = obj.params.step_decay;
            % net = obj.net;
            % subnet = obj.subnet;

            [XC_num,~,~] = size(X);
            [UC_num,num_steps,batch_size] = size(U);
            % extract Koopman net parameters
            subnet.Learnables(1:2,3) = net.Learnables(1:2,3);
            subnet.Learnables(3:4,3) = net.Learnables(end-1:end,3);

            k_loss=0; % k step loss
            aug_loss = 0; % augment loss
            aug_loss2 = 0;
            aug_loss3 = 0;
            aug_loss4 = 0;
            gamma = 1; % weight of each step
            gamma_sum = 0;

            dl_X_i = dlarray(reshape(X(:,1,:),[XC_num,batch_size]), 'CBT');
            dl_U_i = dlarray(reshape(U(:,1,:),[UC_num,batch_size]), 'CBT');
            dl_T_i = dlarray(reshape(T(:,1,:),[XC_num,batch_size]), 'CBT');
            dl_dT_i = dlarray(reshape(dT(:,1,:),[XC_num,batch_size]), 'CBT');
            [z_current,state] = forward(net,dl_U_i, dl_X_i,'Outputs' ,'z');
            z_history(:,:,1) = z_current;
            [ZC_num,batch_size,~] = size(z_current);
            for i = 1:num_steps
                % i=1;
                % decay_rate = 0.9;
                dl_X_i = dlarray(reshape(X(:,i,:),[XC_num,batch_size]), 'CBT');
                dl_U_i = dlarray(reshape(U(:,i,:),[UC_num,batch_size]), 'CBT');
                dl_T_i = dlarray(reshape(T(:,i,:),[XC_num,batch_size]), 'CBT');
                dl_dT_i = dlarray(reshape(dT(:,i,:),[XC_num,batch_size]), 'CBT');

                W2 = (net.Learnables{5,3}{1});
                W1 = (net.Learnables{3,3}{1});

                [phix_T,~] = forward(net,dl_U_i, dl_T_i,'Outputs' ,'phix_k');
                % d_relu = 0*phix_T;
                % d_relu(phix_T>=0) = 1;
                % diag(extractdata(d_relu));
                % Jacob = [eye(XC_num); diag(extractdata(d_relu))*W2*W1];

                [dz_current,~]=forward(subnet,dl_U_i, z_current,'Outputs','dz'); % predect k steps: dz_{k} = Az_k + Bu
                if i<=2
                    z_history(:,:,i+1) = z_history(:,:,i) + Ts*dz_current; % predect k steps: z_{k+1} = z_k + Ts * dz_k
                else
                    z_history(:,:,i+1) = 3/2*z_history(:,:,i) - z_history(:,:,i-1) + 1/2*z_history(:,:,i-2)  + Ts*dz_current;
                end
                z_current = dlarray(reshape(z_history(:,:,i+1),[ZC_num,batch_size]), 'CBT');

                % dz_T = Jacob * reshape(dl_dT_i(:,:,:),[XC_num,batch_size]);

                [z_T,~] = forward(net,dl_U_i, dl_T_i,'Outputs' ,'z'); % lift the target state: phi(x_{k+1})
                [z_current_encoded, state] = forward(net,dl_U_i, z_current(1:XC_num,:,:),'Outputs' ,'z'); % lift predicted state: phi(z_{k+1}(1:num_original_state))

                % k_loss = k_loss + gamma*mse(dz_current, dz_T);
                aug_loss = aug_loss + gamma*mse(z_current, z_current_encoded);
                aug_loss2 = aug_loss2 + gamma*mse(z_current, z_T);
                aug_loss3 = aug_loss3 + gamma*mse(dl_dT_i, dz_current(1:XC_num,:,:));
                aug_loss4 = aug_loss4 + gamma*mse(dl_T_i, z_current(1:XC_num,:,:));

                gamma = gamma * decay_rate;
                gamma_sum = gamma_sum + gamma;
            end

            % Calculate total loss.
            loss = (k_loss + 0.5 * aug_loss +  aug_loss2 + aug_loss3 + aug_loss4) / gamma_sum;
            % loss = mse(z_n, z_T);
            % Calculate gradients of loss with respect to learnable parameters.
            gradients = dlgradient(loss,net.Learnables);

        end


        %% val_model (compares model simulation to real data)
        function results = val_model( obj,  valdata )
            %val_model: Compares a model simulation to real data
            %   liftedSys - struct with fields A, B, C, sys, ...
            %   valdata - struct with fields t, y, u (at least)
            %   results - struct with simulation results and error calculations
            
            % shift real data so delays can be accounted for
            index0 = obj.params.nd + 1;  % index of the first state
            treal = valdata.t(index0 : end);    % start simulation late so delays can be taken into account
            yreal = valdata.y(index0 : end , :);
            dyreal = valdata.dy(index0 : end , :);
            ureal = valdata.u(index0 : end , :);
            dureal = valdata.du(index0 : end , :);
            if obj.params.include_input_delay && ~obj.params.velocity_input_delay
                uireal = valdata.ui(index0 : end , :);
            end
            % [ ~ , zetareal, dzetareal ] = obj.get_zeta( valdata );
            % zreal = zeros( size( dzetareal , 2 ) , obj.params.N );
            % dzreal = zeros( size( dzetareal , 2 ) , obj.params.N );
            % for i = 1 : size( dzetareal , 1 )
            %     Jacob = obj.lift.jacobian(zetareal(i,:)',ureal(i,:)');
            %     zreal(i,:) = obj.lift.full( zetareal(i,:)' );
            %     dzreal(i,:) = Jacob * (dzetareal(i,:)');
            % end
            
            % set initial condition
            valdata_wzeta = obj.get_zeta( valdata );
            zeta0 = valdata_wzeta.zeta(1,:)';    % initial state with delays
            % z0 = obj.lift.full( zeta0 );    % initial lifted state
            
            % simulate lifted linear model
%             [ ysim , tsim , zsim ] = lsim(model.sys, ureal , treal , z0); % Don't use lsim. Might be doing weird stuff
            tsim = treal;
            usim = ureal;
            dusim = dureal;
            ysim = zeros( size( yreal ) ); % preallocate
            dysim = zeros( size( yreal ) ); % preallocate
            ysim(1,:) = yreal(1,:); % initialize
            
            dl_x_current = dlarray(zeta0, 'CBT');
            [XC_num,~] = size(zeta0);
            dl_x_history(:,:,1)= dl_x_current;
            dl_U_i = dlarray(ureal(1,:)', 'CBT');
            [z0,~] = forward(obj.net,dl_U_i, dl_x_current,'Outputs' ,'z');
            
            zsim = zeros( obj.params.lifting_dim ); % preallocate
            dzsim = zeros( obj.params.lifting_dim ); % preallocate
            zsim(1,:) = reshape(extractdata(z0), [1,obj.params.lifting_dim]);        % initialize
            obj.model.A = extractdata(obj.net.Learnables{7,3}{1});
            obj.model.A_bias = extractdata(obj.net.Learnables{8,3}{1});
            obj.model.B = extractdata(obj.net.Learnables{1,3}{1});
            obj.model.B_bias = extractdata(obj.net.Learnables{2,3}{1});
            
            for j = 1 : length(treal)-1
                dl_U_i = dlarray(ureal(j,:)', 'CBT');
                dl_x_current = dlarray(valdata_wzeta.zeta(j,:)', 'CBT');
                [dz,state] = forward(obj.net,dl_U_i, dl_x_current,'Outputs' ,'dz');
                dl_dx_current = dz(1:obj.params.nzeta,:,:);
                % [z_T,state] = forward(net,dl_U_i, dl_T_i,'Outputs' ,'z'); % lift the target state: phi(x_{k+1})
                % zsim(j+1,:) = extractdata(z_n');
                if j<=2
                    dl_x_history(:,:,j+1) = dl_x_history(:,:,j) + (treal(j+1)-treal(j)) * dl_dx_current;
                else
                    dl_x_history(:,:,j+1) = 3/2*dl_x_history(:,:,j) - dl_x_history(:,:,j-1) + 1/2*dl_x_history(:,:,j-2) + (treal(j+1)-treal(j)) * dl_dx_current;
                end
                dl_x_current = dlarray(reshape(dl_x_history(:,:,j+1),[XC_num,1]), 'CBT');
                dysim(j,:) = extractdata(dl_dx_current(1:obj.params.n,:,:))';
                ysim(j+1,:) = extractdata( dl_x_current(1:obj.params.n,:,:))';

                % 
                % dzsim(j,:) = ( obj.model.A * zsim(j,:)' + obj.model.A_bias + obj.model.B * usim(j,:)' + obj.model.A_bias )';
                % if j<=2
                %     zsim(j+1,:) = zsim(j,:) + (treal(j+1)-treal(j)) * dzsim(j,:);
                % else
                %     zsim(j+1,:) = 3/2*zsim(j,:) - zsim(j-1,:) + 1/2*zsim(j-2,:) + (treal(j+1)-treal(j)) * dzsim(j,:);
                % end
                % ysim(j+1,:) = zsim(j+1,1:obj.params.n);
                % dysim(j,:) =dzsim(j,1:obj.params.n);
            end
            
            % save simulations in output struct
            results = struct;
            results.t = treal; 
            results.sim.t = tsim;
            results.sim.u = usim;
            results.sim.y = ysim;
            results.sim.dy = dysim;
            % results.sim.z = zsim;
            % results.sim.dz = dzsim;
            results.real.t = treal;
            results.real.u = ureal;
            results.real.y = yreal;
            results.real.dy = dyreal;
            % results.real.z = zreal;
            % results.real.dz = dzreal;
            
            % save error info (optional, could get rid of this)
            results.error = obj.get_error( results.sim , results.real );
        end
        
 
        % get_error (computes the error between real and simulated data)
        function err = get_error(obj, simdata , realdata )
            %get_error: Computes the error between real and simulated data.
            
            err.abs = abs( simdata.dy - realdata.dy );  % absolute error over time
            err.mean = mean( err.abs , 1 );   % average absolute error over time
            err.rmse = sqrt( sum( (simdata.dy - realdata.dy).^2 , 1 ) ./ length(realdata.t) ); % RMSE (over each state)
            err.nrmse = err.rmse ./ abs( max( realdata.dy ) - min( realdata.dy ) );   % RMSE normalized by total range of real data values
        end
        
        % plot_comparison (plots a comparison between simulation and real data)
        function plot_comparison(obj,  simdata , realdata , figtitle)
            %plot_comparison: plots a comparison between simulation and real data.
            
            % quantify the error
            err = obj.get_error( simdata , realdata );
            
            % create new figure
            if nargin > 3
                figure('NumberTitle', 'off', 'Name', figtitle);  
            else
                figure;
            end
            
            % for i = 1 : obj.params.n
                fontsize=11;
                subplot( obj.params.n , 1 , 1 );
                % ylabel( [ 'dy' , num2str(i) ] );
                ylabel( 'X' , 'FontSize', fontsize, 'interpreter','latex', 'Rotation', pi/2 );
                % title( [ 'RMSE = ' , num2str( err.rmse(i) ) ] );
                ylim([-1,1]);
                hold on;
                plot( realdata.t , realdata.dy( : , 1 ) , '-', 'linewidth',2);
                plot( simdata.t , simdata.dy( : , 1 ) , '--','linewidth',2 );
                set(gca, 'FontSize', fontsize)
                hold off;

                subplot( obj.params.n , 1 , 2 );
                % ylabel( [ 'dy' , num2str(i) ] );
                ylabel( 'Y' , 'FontSize', fontsize, 'interpreter','latex' , 'Rotation', pi/2);
                % title( [ 'RMSE = ' , num2str( err.rmse(i) ) ] );
                ylim([-1,1]);
                hold on;
                plot( realdata.t , realdata.dy( : , 2 ) , '-', 'linewidth',2);
                plot( simdata.t , simdata.dy( : , 2 ) , '--','linewidth',2 );
                set(gca, 'FontSize', fontsize)
                hold off;

                subplot( obj.params.n , 1 , 3 );
                % ylabel( [ 'dy' , num2str(i) ] );
                ylabel( 'Z' , 'FontSize', fontsize, 'interpreter','latex', 'Rotation', pi/2 );
                % title( [ 'RMSE = ' , num2str( err.rmse(i) ) ] );
                ylim([-1,1]);
                hold on;
                plot( realdata.t , realdata.dy( : , 3 ) , '-', 'linewidth',2);
                plot( simdata.t , simdata.dy( : , 3 ) , '--','linewidth',2 );
                set(gca, 'FontSize', fontsize)
                hold off;
                
            % end

            legend({'Real value' , 'Prediction'}, 'FontSize', fontsize, 'NumColumns', 2);
            % dim = obj.params.n;
            % dx_a = realdata.dy;
            % dx_p = simdata.dy;
            % t = realdata.t;
            % save val_results.mat dim dx_a dx_p t
            % 
            % figure
            % Real = realdata.dy;
            % DeepKoopman = simdata.dy;
            % st=stackedplot(Real, DeepKoopman);
            % st.DisplayLabels={'X', 'Y', 'Z'};
            % st.XLabel='Time (s)'



            
%             figure
%             for i = 1 : obj.params.n
%                 subplot( obj.params.n , 1 , i );
%                 ylabel( [ 'y' , num2str(i) ] );
%                 title( [ 'NRMSE = ' , num2str( err.nrmse(i) ) ] );
%                 %                 ylim([-1,1]);
%                 hold on;
%                 plot( realdata.t , realdata.dy( : , i ) , 'b-', 'linewidth',2);
%                 plot( simdata.t , simdata.dy( : , i ) , 'r--','linewidth',2 );
%                 hold off;
%             end
%             legend({'Real' , 'Koopman'});
        end
        
        % valNplot_model (run val_model and plot_comparison for a given model)
        function [ results , err ] = valNplot_model(obj)
            %valNplot_model: run val_model and plot_comparison for a given model
            

            results = cell( size(obj.valdata) );    % store results in a cell array
            err = cell( size(obj.valdata) );    % store error in a cell array
            for i = 1 : length(obj.valdata)
                results{i} = val_model( obj , obj.valdata{i} );
                err{i} = obj.get_error( results{i}.sim , results{i}.real );
                plot_comparison(obj, results{i}.sim , results{i}.real  );
            end
            
            % save (or don't save) sysid class, model, and training data
            saveModel = questdlg( 'Would you like to save this model?' , '' , 'Yes' , 'Not right now' , 'Not right now' );
            if strcmp( saveModel , 'Yes' )
                obj.save_class;
            end
        end
        
 
        %% resample_ref: Resamples a reference trajectory
        function ref_resampled = resample_ref( obj, ref )
            %resample_ref: Resamples a reference trajectory at the system
            % sampling time.
            % ref - struct with fields:
            %   t - time vector
            %   y - trajectory vector
            
            tr = 0 : obj.params.Ts : ref.t(end);
            ref_resampled = interp1( ref.t , ref.y , tr );
        end
        
    end
end





















