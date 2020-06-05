% Defines the utility function in an extended way
function total_utility = find_total_utility(p, q, N, lam, r)
    % The probability you do not match with a critical person given you are critical
    match_w_crit_if_crit = (1 - (p/((N - 1) * q)))^(N * lam - 1);

    % The probability you do not match with a noncritical person given you are noncritical
    match_w_n_crit_if_crit = (1 - (p/((N * lam - 1) * q)))^(N * (1 - lam));

    % The probability you are critical and receive a match
    prob_and_crit = lam * (1 - match_w_crit_if_crit * match_w_n_crit_if_crit);

    % The probability you are not critical and receive a match
    prob_and_n_crit = (1 - lam) * (1 - (1 - (p/((N - 1) * q)))^(N * lam));

    %Total probability you receive a match
    prob_of_match = prob_and_crit + prob_and_n_crit;

    % Time multiplier
    divisor = 1 - (1 - r) * (1 - lam) * (1 - (p/((N - 1) * q)))^(N * lam - 1);

    % Total utility is this this period utility over time multiplier
    total_utility = prob_of_match / divisor;
end

% Generate the symbolic function
syms symb_utility(p, q, N, lam, r);

% Add relevant assumptions
assume(0 < p & p < 1);
assume(0 < q & q < 1);
assume(0 < r & r < 1);
assume(0 < lam & lam < 1);
assume(N, 'positive');

% Generate equation
symb_utility = find_total_utility(p, q, N, lam, r)

% Find derivatives
growth_with_thickness = diff(symb_utility, N)
benefits_of_thickness_with_prob = diff(growth_with_thickness, p)

% Clear the variables to be reused below
clear p q N lam r growth_with_thickness benefits_of_thickness_with_prob;

% This should hopefully be the same as above, but I'm not sure if syms works
% well with user-defined functions as inputs
% Generate the symbolic function
syms utility_long(p, q, N, lam, r);

% Add relevant assumptions
assume(0 < p & p < 1);
assume(0 < q & q < 1);
assume(0 < r & r < 1);
assume(0 < lam & lam < 1);
assume(N, 'positive');

% Generate equation
utility_long(p, q, N, lam, r) = ((lam * (1 - (p/((N - 1) * q)))^(N * lam - 1) * (1 - (p/((N * lam - 1) * q)))^(N * (1 - lam))) + (1 - lam) * (1 - (1 - (p/((N - 1) * q)))^(N * lam))) / (1 - (1 - r) * (1 - lam) * (1 - (p/((N - 1) * q)))^(N * lam - 1))

% Find derivatives
growth_with_thickness_long = diff(utility_long, N)
benefits_of_thickness_with_prob_long = diff(growth_with_thickness, p)

