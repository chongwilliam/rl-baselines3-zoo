"""
    Functions to compute optimal gains given system stiffness 
"""
import math 

""" 
    Compute optimal gains. If gains are beyond limits, then they are saturated in a way that gets maximum wn with gamma in the limits 
"""
def compute_optimal_gains(Ks: float, wn:float, gamma_lim=[0.5, 1], kv_lim=[1, 10], kp_max=[0.1, 0.7], ki_max=[0.1, 3]):

    # set default nominal values for wn and gamma
    wn = 5  # rad/s
    gamma = 1  # critical damping

    # check if wn yields kp and kv under max values 
    real_pole = 10 * wn  # real pole 
    kv = real_pole + 2 * gamma * wn 
    kp = (2 * gamma * wn * real_pole + wn**2) / Ks 
    ki = (real_pole * wn**2) / Ks

    # check kv saturation 

    # kv saturation 
    if kv > kv_max:
        kv = kv_max

    # gain adjustment for kp and ki 

    # # check saturation
    # if kp > kp_max or kv > kv_max:
    #     # Compute limits from the constraints
    #     # wn_limit_kv = 10 * kv_max / (1 + 20 * gamma)
    #     # wn_limit_kp = math.sqrt( (5 * Ks * kp_max) / (gamma + 5))
    #     wn_limit_kv = kv_max / (2 * gamma + 10)
    #     wn_limit_kp = math.sqrt( (kp_max * Ks) / (1 + 20 * gamma))
    #     wn = min(wn_limit_kv, wn_limit_kp)
        
    #     # Recompute 
    #     real_pole = 10 * wn
    #     kv = real_pole + 2 * gamma * wn 
    #     kp = (2 * gamma * wn * real_pole + wn**2) / Ks         
    #     ki = (real_pole * wn**2) / Ks 
    #     return kp, ki, kv  
    # else:
    #     # Pass through 
    #     ki = (real_pole * wn**2) / Ks 
    #     return kp, ki, kv 

import scipy.optimize as opt

def solve_fast_optimization(kv_min, kv_max, kp_min, kp_max, ki_min, ki_max, ke, wn_upper_bound):
    """
    Solve the optimization problem with constraints using an efficient method.

    Parameters:
    - kv_min, kv_max: Limits for the kv constraint
    - kp_min, kp_max: Limits for the kp constraint
    - ki_min, ki_max: Limits for the ki constraint
    - ke: Constant value
    - wn_upper_bound: Maximum bound for wn

    Returns:
    - result.x: Optimal gamma and wn
    - result.fun: Maximum value of wn
    """
    # Objective function: maximize wn
    def objective(x):
        _, wn = x
        return -wn  # Minimize negative wn to maximize wn

    # Constraints
    def kv_constraint(x):
        gamma, wn = x
        return 2 * gamma * wn + 10 * wn 

    def kp_constraint(x):
        gamma, wn = x
        return (wn**2 + 20 * gamma * wn**2) / ke

    def ki_constraint(x):
        _, wn = x
        return (10 * wn**3) / ke

    # Bounds for gamma and wn
    bounds = [(0.5, 1.5), (0, wn_upper_bound)]  # gamma in [0.5, 1], wn in [0, wn_upper_bound]

    # Constraint list
    constraints = [
        {'type': 'ineq', 'fun': lambda x: kv_constraint(x) - kv_min},  # kv >= kv_min
        {'type': 'ineq', 'fun': lambda x: kv_max - kv_constraint(x)},  # kv <= kv_max
        {'type': 'ineq', 'fun': lambda x: kp_constraint(x) - kp_min},  # kp >= kp_min
        {'type': 'ineq', 'fun': lambda x: kp_max - kp_constraint(x)},  # kp <= kp_max
        {'type': 'ineq', 'fun': lambda x: ki_constraint(x) - ki_min},  # ki >= ki_min
        {'type': 'ineq', 'fun': lambda x: ki_max - ki_constraint(x)},  # ki <= ki_max
    ]

    # Initial guess: start near the upper bound of wn
    x0 = [1.0, wn_upper_bound * 0.8]  # gamma = 0.75, wn slightly below the upper bound

    # Solve the optimization problem
    result = opt.minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-4}  # High precision
    )

    if result.success:
        # return result.x, -result.fun
        
        # compute gains 
        return kp_constraint(result.x), ki_constraint(result.x), kv_constraint(result.x)

    else:
        # raise ValueError("Optimization did not converge:", result.message)
        print("Optimization didn't converge")
        return 0, 0, kv_max


# # Example usage
# kv_min = 1.0
# kv_max = 15.0
# kp_min = 0.001
# kp_max = 0.7
# ki_min = 0.001
# ki_max = 2.5
# ke = 10000
# wn_upper_bound = 10

# # kv_min = 5
# # kv_max = 50
# # kp_min = 1
# # kp_max = 10
# # ki_min = 0.1
# # ki_max = 5
# # ke = 2
# # wn_upper_bound = 10

# optimal_values, max_wn = solve_fast_optimization(kv_min, kv_max, kp_min, kp_max, ki_min, ki_max, ke, wn_upper_bound)
# print("Optimal gamma and wn:", optimal_values)
# print("Maximum wn:", max_wn)


# if __name__ == '__main__':
#     kp, ki, kv = compute_optimal_gains(1000, 10)
#     print(kp, ki, kv)