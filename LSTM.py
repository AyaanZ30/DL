import numpy as np               
import pandas as pd              
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/NationalNames.csv')
print(data.columns)

data['Name'] = data['Name']

data = np.array(data['Name'][:10000]).reshape(-1, 1)
data = [name.lower() for name in data[:, 0]]
data = np.array(data).reshape(-1, 1)     # one column having all names

def get_longest_name(data):
    names_length_vector = [len(name[0]) for name in data]
    max_index = names_length_vector.index(max(names_length_vector))
    print("Longest name:", data[max_index][0], f'(Length = {max(names_length_vector)})')
    return max(names_length_vector)

def pad_data(data, longest_name_length):
    for index in range(len(data)):
        rem_len = (longest_name_length - len(data[index, 0]))
        padding_string = '.'*rem_len
        data[index, 0] = data[index, 0] + padding_string
    return data

def build_vocab(data):
    vocab = list()
    for name in data[:, 0]:
        vocab.extend(list(name))
    print(f'Vocab size : {len(set(vocab))}')
    print(f'Vocab : {set(vocab)}')
    return set(vocab), len(vocab)   

transformed_data = np.copy(data)
longest_name_length = get_longest_name(transformed_data)   

transformed_data = pad_data(transformed_data, longest_name_length)
transformed_data[:3]

vocab, vocab_size = build_vocab(transformed_data)

id_char, char_id = dict(), dict()

for i, char in enumerate(vocab):
    id_char[i] = char 
    char_id[char] = i  

print(f'a-{char_id['a']}, 22-{id_char[22]}')

train_dataset = []
batch_size = 20 
TOTAL_BATCHES = int(len(transformed_data) / batch_size)
print(f'Total number of batches in train data : {TOTAL_BATCHES} ({batch_size} samples per batch)')

# splitting train data into batches 
for i in range(len(transformed_data) - batch_size + 1):
    start = i * batch_size
    end = start + batch_size
    
    batch_data = transformed_data[start:end]
    if(len(batch_data) != batch_size):
        break   
    
    char_list = []
    for k in range(len(batch_data[0][0])):
        batch_dataset = np.zeros([batch_size, len(vocab)])
        for j in range(batch_size):
            name = (batch_data[j][0])
            char_index = char_id[name[k]]
            batch_dataset[j, char_index] = 1.0
        
        # one-hot-encoding for ith char of each name in batch_data 
        char_list.append(batch_dataset)
    train_dataset.append(char_list)

print('basically a dataset of 500 samples -> 12 vectors per sample -> 20 ohe vec per vector')

# Hyperparams 
input_units = 100 
hidden_units = 256 
output_units = vocab_size
learning_rate = 5e-3

# for adam optimizer calculations
beta1 = 0.90
beta2 = 0.99
epsilon = 1e-6

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

def tanh_activation(x):
    return np.tanh(x)

def softmax(x):
    exp_x_sum = np.sum(np.exp(x), axis = 1).reshape(-1, 1)
    return (np.exp(x) / exp_x_sum)

def tanh_derivative(x): 
    return (1 - (x ** 2))

def sigmoid_derivative(x):
    return (x * (1 - x))

def initialize_parameters():
    params = dict ()
    mean, std = 0, 0.01 
    
    # LSTM memory cell gates weights 
    input_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    forget_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    output_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    intermediate_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    
    hidden_output_weights = np.random.normal(mean, std, (hidden_units, len(vocab)))
    params['fgw'] = forget_gate_weights
    params['igw'] = input_gate_weights
    params['ogw'] = output_gate_weights 
    params['ggw'] = intermediate_gate_weights 
    params['how'] = hidden_output_weights
    
    return params 
    
    
def lstm_cell(batch_dataset, prev_hidden_state, prev_cell_state, parameters):
    lstm_activations = dict() 
    
    fgw = parameters['fgw']
    ogw = parameters['ogw']
    igw = parameters['igw']
    ggw = parameters['ggw']
    
    # getting current input (from batch dataset) & prev hidden state 
    concat_dataset = np.concatenate((batch_dataset, prev_hidden_state), axis = 1)
    
    fa = sigmoid(np.matmul(concat_dataset, fgw))
    ia = sigmoid(np.matmul(concat_dataset, igw))
    ga = tanh_activation(np.matmul(concat_dataset, ggw))
    oa = sigmoid(np.matmul(concat_dataset, ogw))
    
    # new cell state 
    new_cell_state = np.multiply(prev_cell_state, fa) + np.multiply(ia, ga)
    # new activation
    # new_hidden_state = np.matmul(oa, tanh_activation(new_cell_state))
    new_hidden_state = oa * tanh_activation(new_cell_state)
    
    lstm_activations['fa'] = fa
    lstm_activations['ia'] = ia
    lstm_activations['ga'] = ga
    lstm_activations['oa'] = oa
    return lstm_activations, new_cell_state, new_hidden_state

def output_cell(current_hidden_state, parameters):
    how = parameters['how']
    ot = np.matmul(current_hidden_state, how)
    return softmax(ot)

def get_embeddings(batch_dataset, embeddings):
    embedding_dataset = np.matmul(batch_dataset, embeddings)
    return embedding_dataset

def forward_propagation(batches, parameters, embeddings):
    lstm_cache = dict()                 # for every cell : (fa, ga, oa, ia)
    cell_cache = dict()                 # for every cell : c(t) (cell state)
    activation_cache = dict()           # for every cell : a(t)
    output_cache = dict()               # for every cell : o(t)
    embedding_cache = dict()            # embeddings for each batch : e0, e1,..
    
    batch_size = batches[0].shape[0]
    # initial hidden state(a0) and cell state(c0) 
    a0 = np.zeros([batch_size, hidden_units], dtype = np.float32)
    c0 = np.zeros([batch_size, hidden_units], dtype = np.float32)
    
    activation_cache['a0'] = a0
    cell_cache['c0'] = c0 
    
    for i in range(len(batches) - 1):
        batch_dataset = batches[i]
        
        # instead of using raw & sparse (mostly 0s) OHE vectors, embedding dim matrix is used to represent better information 
        batch_dataset = get_embeddings(batch_dataset, embeddings)
        embedding_cache['emb'+str(i)] = batch_dataset
        
        # get activations and new cell state(ct), new hidden state(ht) for current memory cell
        lstm_activations, ct, at = lstm_cell(batch_dataset, a0, c0, parameters)
        ot = output_cell(at, parameters)
        
        lstm_cache['lstm'+str(i+1)] = lstm_activations  
        output_cache['o'+str(i+1)] = ot
        cell_cache['c'+str(i+1)] = ct
        activation_cache['a'+str(i+1)] = at  
        
        a0, c0 = at, ct      # update for next cell 
    return embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache 

def calculate_loss_accuracy(batch_labels, output_cache):
    acc = 0 
    loss = 0 
    # prob = 1 
    batch_size = batch_labels[0].shape[0]
    for i in range(1, len(output_cache) + 1):
        pred = output_cache['o'+str(i)]
        current_batch_labels = batch_labels[i]
        
        epsilon = 1e-8
        loss += np.sum(
            (current_batch_labels * np.log(pred + epsilon))
            + 
            ((1-current_batch_labels) * np.log(1-pred + epsilon)),
            axis = 1
        ).reshape(-1, 1)
        acc += np.mean(np.argmax(current_batch_labels, axis = 1) == np.argmax(pred, axis = 1))
        
    # perplexity = np.sum((1 / prob)**(1 / len(output_cache))) / batch_size
    loss = np.sum(loss)*(-1 / batch_size)
    acc = acc / len(output_cache)
    
    # return perplexity, loss, acc
    return loss, acc

def output_cell_error(batch_labels, output_cache, parameters):
    output_error_cache = dict()
    activation_error_cache = dict()
    
    for i in range(1, len(output_cache) + 1):
        pred = output_cache['o'+str(i)]
        labels = batch_labels[i]
        
        if pred.shape[1] != labels.shape[1]:
            pred = pred[:, :labels.shape[1]]
        
        # output error for time step 't'
        error_output = (pred - labels)
        
        how = parameters['how']
        if how.shape[1] != labels.shape[1]:
            how = how[:, :labels.shape[1]] 
        # hidden state (activation) error for time step 't'
        error_activation = np.matmul(error_output, how.T)
        
        output_error_cache['eo'+str(i)] = error_output
        activation_error_cache['ea'+str(i)] = error_activation
    return output_error_cache, activation_error_cache

def single_lstm_cell_error(
    activation_output_error,
    next_activation_error,
    next_cell_error,
    parameters,
    lstm_activation,
    cell_activation,
    prev_cell_activation
):
    if next_activation_error.shape[1] != activation_output_error.shape[1]:
        # Pad next_activation_error to match activation_output_error shape
        next_activation_error = np.pad(
            next_activation_error,
            ((0, 0), (0, activation_output_error.shape[1] - next_activation_error.shape[1])),
            'constant'
        )
    # error of hidden state h(t) through output gate
    activation_error = activation_output_error + next_activation_error  
    
    # output gate error (oa) (while bptt) 
    oa = lstm_activation['oa']
    eo = activation_error * tanh_activation(cell_activation) * sigmoid_derivative(oa)
    
    # cell activation error (c(t) error)
    cell_error = (activation_error * oa * tanh_derivative(tanh_activation(cell_activation)))
    cell_error += next_cell_error   # accumulating next cell's error as well in bptt
    
    # input gate error (ia & ga) (while bptt) 
    ia = lstm_activation['ia']
    ga = lstm_activation['ga']
    ei = (cell_error * ga * sigmoid_derivative(ia))
    
    # intermediate gate (ga) error (while bptt)
    eg = cell_error * ia * tanh_derivative(ga)
    
    # forget gate error (fa) (while bptt)
    # prev_cell_activation : activation value that forget gate receiv   ed from prev cell (and decided to retain/discard)
    fa = lstm_activation['fa']
    ef = cell_error * prev_cell_activation * sigmoid_derivative(fa)
    
    # Error to propagate to previous time step's cell state c(t-1)
    prev_cell_error = np.multiply(cell_error, fa)
    
    # getting weights of gates from parameters 
    fgw = parameters['fgw']
    igw = parameters['igw']
    ggw = parameters['ggw']
    ogw = parameters['ogw']
    
    # embedding + hidden activation error 
    embed_activation_error = ef@fgw.T + ei@igw.T + eo@ogw.T + eg@ggw.T 
    
    input_hidden_units = fgw.shape[0]
    hidden_units = fgw.shape[1]
    input_units = input_hidden_units - hidden_units
    
    # prev activation error (Splits error to get portion for previous hidden state h(t-1) & x(t) as embedding error)
    prev_activation_error = embed_activation_error[:, hidden_units:]
    # current input error (x(t))
    embed_error = embed_activation_error[:, :input_units]
    
    lstm_error = dict()
    lstm_error['ef'] = ef 
    lstm_error['ei'] = ei  
    lstm_error['eg'] = eg 
    lstm_error['eo'] = eo   
    
    return prev_activation_error, prev_cell_error, embed_error, lstm_error

def output_cell_derivatives(output_error_cache, activation_cache, parameters):
    derivative_hidden_output_weights = np.zeros_like(parameters['how'])
    
    # the first activation after processing the first batch would be a1 (as a0 would be intiial start)
    batch_size = activation_cache['a1'].shape[0]
    for i in range(1, len(output_error_cache) + 1):
        output_error = output_error_cache['eo'+str(i)]   
        activation = activation_cache['a'+str(i)]
        
        # dhow = (t=1 -> t=T)Σ (activation(t).T * output_error(t)) / batch_size
        derivative_hidden_output_weights += np.matmul(activation.T, output_error) / batch_size
    return derivative_hidden_output_weights

def single_lstm_cell_derivatives(lstm_error, activation_matrix, embedding_matrix):
    # get all errors of LSTM cell for a single time step from cache
    ei = lstm_error['ei']
    ef = lstm_error['ef']
    eo = lstm_error['eo']
    eg = lstm_error['eg']
    
    # get input activations for this time step
    concat_matrix = np.concatenate((activation_matrix, embedding_matrix), axis = 1)
    batch_size = embedding_matrix.shape[0]
    
    dfgw = np.matmul(concat_matrix.T, ef) / batch_size
    digw = np.matmul(concat_matrix.T, ei) / batch_size
    dogw = np.matmul(concat_matrix.T, eo) / batch_size
    dggw = np.matmul(concat_matrix.T, eg) / batch_size
    
    derivatives = dict()
    derivatives['dfgw'] = dfgw  
    derivatives['digw'] = digw 
    derivatives['dogw'] = dogw 
    derivatives['dggw'] = dggw
    
    return derivatives

# The goal is to compute gradients for all weights using the chain rule considering LSTM-specific gates and memory cells.
def backward_propagation(
    parameters, 
    batch_labels,
    embedding_cache, 
    lstm_cache, 
    activation_cache, 
    cell_cache, 
    output_cache
):
    output_errors_cache, activation_errors_cache = output_cell_error(batch_labels, output_cache, parameters)
    # to store lstm error for each time step
    lstm_error_cache = dict()
    # to store embeddings (input) error for each time step
    embedding_error_cache = dict()
    
    # At the last time step (t=T), there are no future errors to propagate    
    error_activation_at_t = np.zeros(activation_cache['a1'].shape)
    error_cellstate_at_t = np.zeros(cell_cache['c1'].shape)
    
    # loop in reverse order (from t = T -> t = 0)
    for i in range(len(lstm_cache), 0, -1): 
        prev_activation_error, prev_cell_error, embed_error, lstm_error = single_lstm_cell_error(
            activation_output_error = activation_cache['a'+str(i)],
            next_activation_error = error_activation_at_t,
            next_cell_error = error_cellstate_at_t, 
            parameters = parameters, 
            lstm_activation = lstm_cache['lstm'+str(i)],  
            cell_activation = cell_cache['c'+str(i)], 
            prev_cell_activation = cell_cache['c'+str(i-1)]
        )
        lstm_error_cache['elstm'+str(i)] = lstm_error 
        embedding_error_cache['eemb'+str(i)] = embed_error
        
        # now we go for next (t-1) so prev would be at t
        error_activation_at_t = prev_activation_error
        error_cellstate_at_t = prev_cell_error
    
    # calculating output cell derivative (gradient) 
    derivatives = dict()
    # derivative_hidden_output_weights -> derivatives ['dhow']
    derivatives['dhow'] = output_cell_derivatives(
        output_error_cache = output_errors_cache, 
        activation_cache = activation_cache,
        parameters = parameters 
    )
    
    # caiculatign lstm cell derivatives at each time step and storing in dict
    lstm_derivatives = dict()
    for i in range(1, len(lstm_error_cache) + 1):
        lstm_derivatives['dlstm'+str(i)] = single_lstm_cell_derivatives(
            lstm_error = lstm_error_cache['elstm'+str(i)], 
            activation_matrix = activation_cache['a'+str(i-1)], 
            embedding_matrix = embedding_cache['emb'+str(i-1)]
        )
        
    #initialize the derivatives to zeros 
    derivatives['dfgw'] = np.zeros(parameters['fgw'].shape)
    derivatives['digw'] = np.zeros(parameters['igw'].shape)
    derivatives['dogw'] = np.zeros(parameters['ogw'].shape)
    derivatives['dggw'] = np.zeros(parameters['ggw'].shape)
    
    # sum up derivatives for each time step (all 4 gates weights independently for each lstm cell)
    for i in range(1, len(lstm_error_cache) + 1):
        derivatives['dfgw'] += lstm_derivatives['dlstm'+str(i)]['dfgw']
        derivatives['digw'] += lstm_derivatives['dlstm'+str(i)]['digw']
        derivatives['dogw'] += lstm_derivatives['dlstm'+str(i)]['dogw']
        derivatives['dggw'] += lstm_derivatives['dlstm'+str(i)]['dggw']
    
    return derivatives, embedding_error_cache

def update_params_adam_optimization(parameters, derivatives, V, S):
    # get the derivatives
    dfgw = derivatives['dfgw']
    digw = derivatives['digw']
    dogw = derivatives['dogw']
    dggw = derivatives['dggw']
    dhow = derivatives['dhow']
    
    # get parameters 
    fgw = parameters['fgw']
    igw = parameters['igw']
    ogw = parameters['ogw']
    ggw = parameters['ggw']
    how = parameters['how']
    
    # get V (MOMENTUM) parameters (as learning rate is to be optimized for each parameter)
    vfgw = V['vfgw']
    vigw = V['vigw']
    vogw = V['vogw']
    vggw = V['vggw']
    vhow = V['vhow']
    
    # get S (RMS-PROP) parameters (as learning rate is to be optimized for each parameter)
    sfgw = S['sfgw']
    sigw = S['sigw']
    sogw = S['sogw']
    sggw = S['sggw']
    show = S['show']
    
    # calculate momentum aspect for the moving of gradients
    vfgw = (beta1 * vfgw) + (1-beta1)*(dfgw)
    vigw = (beta1 * vigw) + (1-beta1)*(digw)
    vogw = (beta1 * vogw) + (1-beta1)*(dogw)
    vggw = (beta1 * vggw) + (1-beta1)*(dggw)
    vhow = (beta1 * vhow) + (1-beta1)*(dhow)
    
    # calculate the RMS aspect using square of gradients
    sfgw = (beta2 * sfgw) + (1-beta2)*(dfgw**2)
    sigw = (beta2 * sigw) + (1-beta2)*(digw**2)
    sogw = (beta2 * sogw) + (1-beta2)*(dogw**2)
    sggw = (beta2 * sggw) + (1-beta2)*(dggw**2)
    show = (beta2 * show) + (1-beta2)*(dhow**2)
    
    # FINALLY, UPDATE WEIGHTS USING FORMULA IN THE PICTURE
    fgw = fgw - (learning_rate)*((vfgw)/np.sqrt(sfgw) + epsilon)
    igw = igw - (learning_rate)*((vigw)/np.sqrt(sigw) + epsilon)
    ogw = ogw - (learning_rate)*((vogw)/np.sqrt(sogw) + epsilon)
    ggw = ggw - (learning_rate)*((vggw)/np.sqrt(sggw) + epsilon)
    how = how - (learning_rate)*((vhow)/np.sqrt(show) + epsilon)
    
    # store the updated weights by replacing initial ones
    parameters['fgw'] = fgw 
    parameters['igw'] = igw 
    parameters['ogw'] = ogw 
    parameters['ggw'] = ggw 
    parameters['how'] = how 
    
    # store updated V params    
    V['vfgw'] = vfgw
    V['vigw'] = vigw 
    V['vogw'] = vogw 
    V['vggw'] = vggw
    V['vhow'] = vhow
    
    #store updated S parameters
    S['sfgw'] = sfgw 
    S['sigw'] = sigw 
    S['sogw'] = sogw 
    S['sggw'] = sggw
    S['show'] = show
    
    return parameters, V, S

def initialize_V(parameters):
    V = dict()
    V['vfgw'] = np.zeros(parameters['fgw'].shape)
    V['vigw'] = np.zeros(parameters['igw'].shape)
    V['vogw'] = np.zeros(parameters['ogw'].shape)
    V['vggw'] = np.zeros(parameters['ggw'].shape)
    V['vhow'] = np.zeros((hidden_units, len(vocab)))
    return V

def initialize_S(parameters):
    S = dict()
    S['sfgw'] = np.zeros(parameters['fgw'].shape)
    S['sigw'] = np.zeros(parameters['igw'].shape)
    S['sogw'] = np.zeros(parameters['ogw'].shape)
    S['sggw'] = np.zeros(parameters['ggw'].shape)
    S['show'] = np.zeros((hidden_units, len(vocab)))  # Use vocab size
    return S

def update_embeddings(embeddings, embedding_error_cache, batch_labels):
    embedding_derivatives = np.zeros(embeddings.shape)
    batch_size = batch_labels[0].shape[0]
    
    for i in range(len(embedding_error_cache)):
        embedding_derivatives += np.matmul(
            batch_labels[i].T, 
            embedding_error_cache['eemb'+str(i+1)]
        ) / (batch_size)
    
    embeddings = embeddings - (learning_rate) * (embedding_derivatives)
    return embeddings

def train_lstm(train_dataset, iterations = 8000, batch_size = 20):
    print('Train dataset (no of batchesm) : ',len(train_dataset))
    parameters = initialize_parameters()
    
    V = initialize_V(parameters)
    S = initialize_S(parameters)
    
    # initialize embeddings of shape (27 x 100) => (vocab size x input units)
    embeddings = np.random.normal(0, 0.01, (len(vocab), input_units))
    J,A = [], []               # loss, perplexity, accuracy
    
    for i in range(iterations):
        index = i % len(train_dataset)
        batches = train_dataset[index]
        
        # FP
        embedding_cache,lstm_cache,activation_cache,cell_cache,output_cache = forward_propagation(
            batches = batches,
            parameters = parameters, 
            embeddings = embeddings
        )
        
        # Metrics
        # perplexity, loss, accuracy = calculate_loss_accuracy(batches, output_cache)
        loss, accuracy = calculate_loss_accuracy(batches, output_cache)
        
        # BPTT 
        derivatives, embedding_error_cache = backward_propagation(
            parameters = parameters, 
            batch_labels = batches, 
            embedding_cache = embedding_cache,
            lstm_cache = lstm_cache,
            activation_cache = activation_cache, 
            cell_cache = cell_cache, 
            output_cache = output_cache  
        )
        
        # updating params using adam
        parameters, V, S = update_params_adam_optimization(
            parameters = parameters, 
            derivatives = derivatives,
            V = V, 
            S = S
        )
        
         # updating embeddings
        embeddings = update_embeddings(
            embeddings = embeddings,
            embedding_error_cache = embedding_error_cache, 
            batch_labels = batches
        )
        
        J.append(loss)
        A.append(accuracy)
        
        if(i%1000==0):
            print("For Single Batch :")
            print('Step       = {}'.format(i))
            print('Loss       = {}'.format(round(loss,2)))
            # print('Perplexity = {}'.format(round(perplexity,2)))
            print('Accuracy   = {}'.format(round(accuracy*100,2)))
            print()
        
    return embeddings, parameters, J, A
    
embeddings, parameters, J, A = train_lstm(train_dataset = train_dataset)
    