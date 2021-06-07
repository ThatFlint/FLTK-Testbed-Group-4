'''Update the distribution of configurations and calculate the entropy'''

import math


# Inputs: 1. dist, the distribution of configurations
# 2. configs, the possible configurations
# 3. chosen_configs, the configurations chosen by clients in one epoch
# 4. losses: loss sent by each client
# 5. V: size of the validation data size of each client
def update_dist(dist, configs, chosen_configs, losses, V):
    new_dist = []
    for p in dist:
        new_dist.append(p)
    learning_rate = math.sqrt(2*math.log10(len(configs))) # Learning rate for the distribution
    factor = 0.005    # A multiplying factor to reduce the variance of grads
    grads = [0]*len(configs)

    # Calculate the sum of the length of the validation set
    sum_v = 0
    for v in V:
        sum_v += v

    # Calculate the sum of the losses
    sum_loss = 0
    for loss in losses:
        sum_loss += loss

    # Calculate the baseline to reduce the variance of grads
    lambda_t = sum_loss/sum_v

    # Calculate gradient
    for j in range(len(configs)):
        for i in range(len(chosen_configs)):
            if chosen_configs[i] == configs[j]:
                grads[j] += factor*losses[i]*V[i]/(dist[j]*sum_v)
                # grads[j] += (losses[i]-lambda_t)*V[i] / (dist[j]*sum_v)

    # Update distribution
    sum_p = 0
    for j in range(len(dist)):
        new_dist[j] *= math.exp(-learning_rate*grads[j])
        sum_p += new_dist[j]

    # Normalization
    for j in range(len(dist)):
        new_dist[j] = new_dist[j]/sum_p
    return new_dist

def cal_dist_entropy(dist):
    entropy = 0.0
    for i in range(len(dist)):
        entropy += -dist[i]*math.log(dist[i])
    return entropy


if __name__ == "__main__":
    dist = [0.2, 0.2, 0.2, 0.2, 0.2]
    configs = [10, 16, 32, 64, 128]
    chosen_configs = [10, 128, 64, 64, 16]
    losses = [90, 100, 85, 79, 60]
    V = [10, 10, 10, 10, 10]
    new_dist = update_dist(dist, configs, chosen_configs, losses, V)
    old_entropy = cal_dist_entropy(dist)
    new_entropy = cal_dist_entropy(new_dist)
    print(f"dist: {dist}")
    print(f"New dist: {new_dist}")
    print(f"Old entropy: {old_entropy}\nNew entropy: {new_entropy}")

