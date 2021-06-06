def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        # Here parameters is a list of weights. parameters = [client1_weights, client2_weights...]
        # client1_weights is a dictionary
        # len(parameters) equals the number of clients
        # In this method every client is equally important
        new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

    return new_params

def fed_average_nn_parameters(parameters, sizes):
    # In this method the importance of a client is determined by its training data size
    new_params = {}
    sum_size = 0
    for client in range(len(parameters)):
        for name in parameters[client].keys():
            try:
                new_params[name].data += (parameters[client][name].data * sizes[client])
            except: 
                new_params[name] = (parameters[client][name].data * sizes[client])
        sum_size += sizes[client]

    for name in new_params:
        # new_params[name].data /= sum_size
        new_params[name].data = new_params[name].data/sum_size

    return new_params