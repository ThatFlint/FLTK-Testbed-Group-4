'''Update the distribution of configurations and calculate the entropy'''

import math


def update_dist(dist, configs, chosen_configs, losses, V, max_grads):
    # 1. dist: the distribution of configurations
    # 2. configs: the possible configurations
    # 3. chosen_configs: the configurations chosen by clients in one communication round
    # 4. losses: loss sent by each client
    # 5. V: size of the validation data for each client
    # 6. max_grads: the maximum gradients w.r.t. the probability of configurations in the former rounds
    
    new_dist = [p for p in dist]
    factor = 0.005    # A multiplying factor to reduce the variance of grads
    sum_v = sum(V)
    sum_loss = sum(losses) 
    lambda_t = sum_loss/sum_v # A baseline to reduce the variance of the gradients 

    # Calculate gradient and obtain the maximum gradient
    grads = [0]*len(configs)
    for j in range(len(configs)):
        for i in range(len(chosen_configs)):
            if chosen_configs[i] == configs[j]:
                # grads[j] += factor*losses[i]*V[i]/(dist[j]*sum_v)
                grads[j] += (losses[i]-lambda_t)*V[i] / (dist[j]*sum_v)
    max_grad = max(grads)
    max_grads.append(max_grad)

    # Calculate the learning rate of the probabilities
    # Type 1: constant learning rate
    dist_lr1 = math.sqrt(2*math.log10(len(configs)))/200
    # Type 2: adaptive (decaying) learning rate
    dist_lr2 = dist_lr1/math.sqrt(sum([grad**2 for grad in max_grads]))
    # Type 3: aggressive learning rate
    dist_lr3 = dist_lr1/max_grad

    # Update distribution
    for j in range(len(dist)):
        new_dist[j] *= math.exp(-dist_lr3*grads[j])

    # Normalization
    sum_p = sum(new_dist)
    new_dist = [p/sum_p for p in new_dist]
    return new_dist, max_grads

def cal_dist_entropy(dist):
    entropy = 0.0
    for i in range(len(dist)):
        entropy += -1*dist[i]*math.log(dist[i])
    return entropy


if __name__ == "__main__":
    dist = [0.2, 0.2, 0.2, 0.2, 0.2]
    configs = [10, 16, 32, 64, 128]
    chosen_configs = [10, 128, 64, 64, 16]
    losses = [90, 100, 85, 79, 60]
    V = [10, 10, 10, 10, 10]
    max_grads = [100]
    new_dist, max_grads = update_dist(dist, configs, chosen_configs, losses, V, max_grads)
    print(new_dist)
    entropy = cal_dist_entropy(new_dist)


