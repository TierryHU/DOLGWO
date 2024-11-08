function [best_val, best_x] = DOLGWO(Gm_o, D, Np, lb, ub, fobj, func_num)
% DOLGWO: Dynamic Opposition-based Learning Grey Wolf Optimization
% Optimizes a given objective function using the enhanced GWO algorithm
% Gm_o: Max generations, D: Dimensions, Np: Population size
% lb: Lower bounds, ub: Upper bounds, fobj: Objective function
% func_num: Function number for the objective function

disp('DOLGWO Optimization Running...')

% Initialize boundaries
Lowerbound = ones(1, D) .* lb;
Upperbound = ones(1, D) .* ub;
pop = repmat(Lowerbound, Np, 1) + rand(Np, D) .* (repmat(Upperbound, Np, 1) - repmat(Lowerbound, Np, 1));

% ************ Initialization ************
% Initialize Alpha, Beta, and Delta positions and scores
Alpha_pos = zeros(1, D);
Alpha_score = inf; % Use -inf for maximization problems
Beta_pos = zeros(1, D);
Beta_score = inf;
Delta_pos = zeros(1, D);
Delta_score = inf;

% Recorders for the optimization process
Positions = pop;
best_val = zeros(1, Gm_o);
best_x = zeros(Gm_o, D);
Jr = 0.3; % Jumping rate for dynamic opposition
w = 8; % Weight for opposition learning
Objective_values = zeros(1, Np);

% Initial fitness calculation
for i = 1:Np
    Objective_values(i) = fobj(pop(i, :), func_num); % Fitness value for each wolf
    if Objective_values(i) < Alpha_score
        Alpha_score = Objective_values(i);
        Alpha_pos = pop(i, :); % Update Alpha
    elseif Objective_values(i) < Beta_score
        Beta_score = Objective_values(i);
        Beta_pos = pop(i, :); % Update Beta
    elseif Objective_values(i) < Delta_score
        Delta_score = Objective_values(i);
        Delta_pos = pop(i, :); % Update Delta
    end
end

% ************ GWO Main Loop ************
for G = 1:Gm_o
    % Update the coefficient a linearly from 2 to 0
    a = 2 - G * (2 / Gm_o);
    
    for i = 1:Np
        for j = 1:D
            % Calculate coefficients for the position update
            r1 = rand();
            r2 = (2 * pi) * rand();
            A1 = 2 * a * r1 - a;
            C1 = 2 * rand;
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
            X1 = Alpha_pos(j) - A1 * D_alpha;
            
            r1 = rand();
            A2 = 2 * a * r1 - a;
            C2 = 2 * rand;
            D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
            X2 = Beta_pos(j) - A2 * D_beta;
            
            r1 = rand();
            A3 = 2 * a * r1 - a;
            C3 = 2 * rand;
            D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
            X3 = Delta_pos(j) - A3 * D_delta;
            
            % Update Position
            Positions(i, j) = (X1 + X2 + X3) / 3;
        end
    end
    
    % Boundary check for updated positions
    Positions = Checkbound(Positions, Lowerbound, Upperbound, Np, D);
    
    % Fitness evaluation and update of Alpha, Beta, Delta
    for i = 1:Np
        fitness = fobj(Positions(i, :), func_num);
        if fitness < Alpha_score
            Alpha_score = fitness;
            Alpha_pos = Positions(i, :);
        elseif fitness < Beta_score
            Beta_score = fitness;
            Beta_pos = Positions(i, :);
        elseif fitness < Delta_score
            Delta_score = fitness;
            Delta_pos = Positions(i, :);
        end
    end
    
    % Record the best fitness value for this generation
    best_val(G) = Alpha_score;
    best_x(G, :) = Alpha_pos;
    
    % ************ Dynamic Opposition Learning ************
    if rand < Jr
        for i = 1:Np
            % Generate dynamically opposed solutions
            for j = 1:D
                Upperbound1(j) = max(Positions(:, j));
                Lowerbound1(j) = min(Positions(:, j));
                op(i, j) = Upperbound1(j) + Lowerbound1(j) - Positions(i, j);
            end
            Positions(Np + i, :) = Positions(i, :) + w * rand * (rand * op(i, :) - Positions(i, :));
        end
        
        % Boundary check for dynamically opposed positions
        Positions = Checkbound(Positions, Lowerbound, Upperbound, 2 * Np, D);
        
        % Evaluate dynamically opposed positions and update if necessary
        for i = 1:2 * Np
            fitness = fobj(Positions(i, :), func_num);
            if fitness < Alpha_score
                Alpha_score = fitness;
                Alpha_pos = Positions(i, :);
            elseif fitness < Beta_score
                Beta_score = fitness;
                Beta_pos = Positions(i, :);
            elseif fitness < Delta_score
                Delta_score = fitness;
                Delta_pos = Positions(i, :);
            end
        end
    end
end

% Final best solution
best_val = Alpha_score;
best_x = Alpha_pos;
end

function positions = Checkbound(positions, lower, upper, Np, D)
    % Check and enforce boundary constraints
    for i = 1:Np
        for j = 1:D
            if positions(i, j) < lower(j)
                positions(i, j) = lower(j);
            elseif positions(i, j) > upper(j)
                positions(i, j) = upper(j);
            end
        end
    end
end
