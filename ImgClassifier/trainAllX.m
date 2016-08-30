input_layer_size  = w*d;  % wxd Input Images of Digits
hidden_layer_size = 3200;   %  hidden units
num_labels = 1;  

options = optimset('MaxIter', 700);

lambda = 2;

fprintf('\nInitializing Neural Network_1 Parameters ...\n');

initial_Theta1_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta1_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params1 = [initial_Theta1_1(:) ; initial_Theta1_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X1, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params1, cost] = fmincg(costFunction, initial_nn_params1, options);

Theta1_1 = reshape(nn_params1(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta1_2 = reshape(nn_params1((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

printf('\nInitializing Neural Network_2 Parameters ...\n');

initial_Theta2_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params2 = [initial_Theta2_1(:) ; initial_Theta2_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X2, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params2, cost] = fmincg(costFunction, initial_nn_params2, options);

Theta2_1 = reshape(nn_params2(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2_2 = reshape(nn_params2((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_3 Parameters ...\n');

initial_Theta3_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta3_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params3 = [initial_Theta3_1(:) ; initial_Theta3_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X3, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params3, cost] = fmincg(costFunction, initial_nn_params3, options);

Theta3_1 = reshape(nn_params3(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta3_2 = reshape(nn_params3((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_4 Parameters ...\n');

initial_Theta4_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta4_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params4 = [initial_Theta4_1(:) ; initial_Theta4_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X4, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params4, cost] = fmincg(costFunction, initial_nn_params4, options);

Theta4_1 = reshape(nn_params4(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta4_2 = reshape(nn_params4((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
fprintf('\nInitializing Neural Network_5 Parameters ...\n');

initial_Theta5_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta5_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params5 = [initial_Theta5_1(:) ; initial_Theta5_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X5, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params5, cost] = fmincg(costFunction, initial_nn_params5, options);

Theta5_1 = reshape(nn_params5(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta5_2 = reshape(nn_params5((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_6 Parameters ...\n');

initial_Theta6_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta6_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params6 = [initial_Theta6_1(:) ; initial_Theta6_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X6, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params6, cost] = fmincg(costFunction, initial_nn_params6, options);

Theta6_1 = reshape(nn_params6(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta6_2 = reshape(nn_params6((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_7 Parameters ...\n');

initial_Theta7_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta7_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params7 = [initial_Theta7_1(:) ; initial_Theta7_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X7, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params7, cost] = fmincg(costFunction, initial_nn_params7, options);

Theta7_1 = reshape(nn_params7(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta7_2 = reshape(nn_params7((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_8 Parameters ...\n');

initial_Theta8_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta8_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params8 = [initial_Theta8_1(:) ; initial_Theta8_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X8, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params8, cost] = fmincg(costFunction, initial_nn_params8, options);

Theta8_1 = reshape(nn_params8(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta8_2 = reshape(nn_params8((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_9 Parameters ...\n');

initial_Theta9_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta9_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params9 = [initial_Theta9_1(:) ; initial_Theta9_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X9, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params9, cost] = fmincg(costFunction, initial_nn_params9, options);

Theta9_1 = reshape(nn_params9(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta9_2 = reshape(nn_params9((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_10 Parameters ...\n');

initial_Theta10_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta10_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params10 = [initial_Theta10_1(:) ; initial_Theta10_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X10, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params10, cost] = fmincg(costFunction, initial_nn_params10, options);

Theta10_1 = reshape(nn_params10(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta10_2 = reshape(nn_params10((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_11 Parameters ...\n');

initial_Theta11_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta11_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params11 = [initial_Theta11_1(:) ; initial_Theta11_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X11, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params11, cost] = fmincg(costFunction, initial_nn_params11, options);

Theta11_1 = reshape(nn_params11(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta11_2 = reshape(nn_params11((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_12 Parameters ...\n');

initial_Theta12_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta12_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params12 = [initial_Theta12_1(:) ; initial_Theta12_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X12, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params12, cost] = fmincg(costFunction, initial_nn_params12, options);

Theta12_1 = reshape(nn_params12(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta12_2 = reshape(nn_params12((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
fprintf('\nInitializing Neural Network_13 Parameters ...\n');

initial_Theta13_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta13_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params13 = [initial_Theta13_1(:) ; initial_Theta13_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X13, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params13, cost] = fmincg(costFunction, initial_nn_params13, options);

Theta13_1 = reshape(nn_params13(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta13_2 = reshape(nn_params13((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_14 Parameters ...\n');

initial_Theta14_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta14_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params14 = [initial_Theta14_1(:) ; initial_Theta14_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X14, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params14, cost] = fmincg(costFunction, initial_nn_params14, options);

Theta14_1 = reshape(nn_params14(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta14_2 = reshape(nn_params14((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_15 Parameters ...\n');

initial_Theta15_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta15_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params15 = [initial_Theta15_1(:) ; initial_Theta15_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X15, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params15, cost] = fmincg(costFunction, initial_nn_params15, options);

Theta15_1 = reshape(nn_params15(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta15_2 = reshape(nn_params15((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_16 Parameters ...\n');

initial_Theta16_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta16_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params16 = [initial_Theta16_1(:) ; initial_Theta16_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X16, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params16, cost] = fmincg(costFunction, initial_nn_params16, options);

Theta16_1 = reshape(nn_params16(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta16_2 = reshape(nn_params16((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%end

fprintf('\nInitializing Neural Network_17 Parameters ...\n');

initial_Theta17_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta17_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params17 = [initial_Theta17_1(:) ; initial_Theta17_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X17, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params17, cost] = fmincg(costFunction, initial_nn_params17, options);

Theta17_1 = reshape(nn_params17(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta17_2 = reshape(nn_params17((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

printf('\nInitializing Neural Network_18 Parameters ...\n');

initial_Theta18_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta18_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params18 = [initial_Theta18_1(:) ; initial_Theta18_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X18, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params18, cost] = fmincg(costFunction, initial_nn_params18, options);

Theta18_1 = reshape(nn_params18(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta18_2 = reshape(nn_params18((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_19 Parameters ...\n');

initial_Theta19_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta19_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params19 = [initial_Theta19_1(:) ; initial_Theta19_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X19, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params19, cost] = fmincg(costFunction, initial_nn_params19, options);

Theta19_1 = reshape(nn_params19(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta19_2 = reshape(nn_params19((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_20 Parameters ...\n');

initial_Theta20_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta20_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params20 = [initial_Theta20_1(:) ; initial_Theta20_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X20, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params20, cost] = fmincg(costFunction, initial_nn_params20, options);

Theta20_1 = reshape(nn_params20(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta20_2 = reshape(nn_params20((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
fprintf('\nInitializing Neural Network_21 Parameters ...\n');

initial_Theta21_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta21_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params21 = [initial_Theta21_1(:) ; initial_Theta21_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X21, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params21, cost] = fmincg(costFunction, initial_nn_params21, options);

Theta21_1 = reshape(nn_params21(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta21_2 = reshape(nn_params21((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_22 Parameters ...\n');

initial_Theta22_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta22_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params22 = [initial_Theta22_1(:) ; initial_Theta22_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X22, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params22, cost] = fmincg(costFunction, initial_nn_params22, options);

Theta22_1 = reshape(nn_params22(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta22_2 = reshape(nn_params22((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_23 Parameters ...\n');

initial_Theta23_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta23_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params23 = [initial_Theta23_1(:) ; initial_Theta23_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X23, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params23, cost] = fmincg(costFunction, initial_nn_params23, options);

Theta23_1 = reshape(nn_params23(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta23_2 = reshape(nn_params23((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_24 Parameters ...\n');

initial_Theta24_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta24_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params24 = [initial_Theta24_1(:) ; initial_Theta24_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X24, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params24, cost] = fmincg(costFunction, initial_nn_params24, options);

Theta24_1 = reshape(nn_params24(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta24_2 = reshape(nn_params24((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_25 Parameters ...\n');

initial_Theta25_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta25_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params25 = [initial_Theta25_1(:) ; initial_Theta25_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X25, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params25, cost] = fmincg(costFunction, initial_nn_params25, options);

Theta25_1 = reshape(nn_params25(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta25_2 = reshape(nn_params25((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_26 Parameters ...\n');

initial_Theta26_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta26_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params26 = [initial_Theta26_1(:) ; initial_Theta26_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X26, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params26, cost] = fmincg(costFunction, initial_nn_params26, options);

Theta26_1 = reshape(nn_params26(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta26_2 = reshape(nn_params26((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_27 Parameters ...\n');

initial_Theta27_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta27_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params27 = [initial_Theta27_1(:) ; initial_Theta27_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X27, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params27, cost] = fmincg(costFunction, initial_nn_params27, options);

Theta27_1 = reshape(nn_params27(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta27_2 = reshape(nn_params27((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_28 Parameters ...\n');

initial_Theta28_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta28_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params28 = [initial_Theta28_1(:) ; initial_Theta28_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X28, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params28, cost] = fmincg(costFunction, initial_nn_params28, options);

Theta28_1 = reshape(nn_params28(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta28_2 = reshape(nn_params28((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
fprintf('\nInitializing Neural Network_29 Parameters ...\n');

initial_Theta29_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta29_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params29 = [initial_Theta29_1(:) ; initial_Theta29_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X29, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params29, cost] = fmincg(costFunction, initial_nn_params29, options);

Theta29_1 = reshape(nn_params29(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta29_2 = reshape(nn_params29((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_30 Parameters ...\n');

initial_Theta30_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta30_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params30 = [initial_Theta30_1(:) ; initial_Theta30_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X30, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params30, cost] = fmincg(costFunction, initial_nn_params30, options);

Theta30_1 = reshape(nn_params30(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta30_2 = reshape(nn_params30((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_31 Parameters ...\n');

initial_Theta31_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta31_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params31 = [initial_Theta31_1(:) ; initial_Theta31_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X31, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params31, cost] = fmincg(costFunction, initial_nn_params31, options);

Theta31_1 = reshape(nn_params31(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta31_2 = reshape(nn_params31((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_32 Parameters ...\n');

initial_Theta32_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta32_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params32 = [initial_Theta32_1(:) ; initial_Theta32_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X32, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params32, cost] = fmincg(costFunction, initial_nn_params32, options);

Theta32_1 = reshape(nn_params32(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta32_2 = reshape(nn_params32((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%end


fprintf('\nInitializing Neural Network_33 Parameters ...\n');

initial_Theta33_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta33_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params33 = [initial_Theta33_1(:) ; initial_Theta33_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X33, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params33, cost] = fmincg(costFunction, initial_nn_params33, options);

Theta33_1 = reshape(nn_params33(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta33_2 = reshape(nn_params33((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_34 Parameters ...\n');

initial_Theta34_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta34_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params34 = [initial_Theta34_1(:) ; initial_Theta34_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X34, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params34, cost] = fmincg(costFunction, initial_nn_params34, options);

Theta34_1 = reshape(nn_params34(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta34_2 = reshape(nn_params34((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_35 Parameters ...\n');

initial_Theta35_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta35_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params35 = [initial_Theta35_1(:) ; initial_Theta35_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X35, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params35, cost] = fmincg(costFunction, initial_nn_params35, options);

Theta35_1 = reshape(nn_params35(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta35_2 = reshape(nn_params35((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_36 Parameters ...\n');

initial_Theta36_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta36_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params36 = [initial_Theta36_1(:) ; initial_Theta36_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X36, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params36, cost] = fmincg(costFunction, initial_nn_params36, options);

Theta36_1 = reshape(nn_params36(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta36_2 = reshape(nn_params36((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_37 Parameters ...\n');

initial_Theta37_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta37_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params37 = [initial_Theta37_1(:) ; initial_Theta37_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X37, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params37, cost] = fmincg(costFunction, initial_nn_params37, options);

Theta37_1 = reshape(nn_params37(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta37_2 = reshape(nn_params37((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nInitializing Neural Network_38 Parameters ...\n');

initial_Theta38_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta38_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params38 = [initial_Theta38_1(:) ; initial_Theta38_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X38, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params38, cost] = fmincg(costFunction, initial_nn_params38, options);

Theta38_1 = reshape(nn_params38(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta38_2 = reshape(nn_params38((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
fprintf('\nInitializing Neural Network_39 Parameters ...\n');

initial_Theta39_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta39_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params39 = [initial_Theta39_1(:) ; initial_Theta39_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X39, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params39, cost] = fmincg(costFunction, initial_nn_params39, options);

Theta39_1 = reshape(nn_params39(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta39_2 = reshape(nn_params39((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('\nInitializing Neural Network_40 Parameters ...\n');

initial_Theta40_1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta40_2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params40 = [initial_Theta40_1(:) ; initial_Theta40_2(:)];

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X40, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params40, cost] = fmincg(costFunction, initial_nn_params40, options);

Theta40_1 = reshape(nn_params40(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta40_2 = reshape(nn_params40((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%ende

fprintf('Program paused. Press enter to continue.\n');
pause;