import jax.numpy as jnp
import jax
import numpy as np
from scipy import optimize

class SNMF():
    def __init__(self, n_input_neurons, n_gate_neurons, decay1=0.99, decay2=0.99) -> None:
        """
        Online clustering by Symmetric Non-negative Matrix Factorization Algorithm, from 10.1109/ACSSC.2014.7094553.
        decay1: used to decay Y;
        decay2: used to decay the weghts from gate neurons to hidden layer.
        """
        self.n_input_neurons =n_input_neurons
        self.n_gate_neurons = n_gate_neurons
        self.decay1, self.decay2 = decay1, decay2
        self.W = np.random.normal(scale=0.01, size=(n_gate_neurons, n_input_neurons)) # Hebbian weights from input layer to gate layer
        self.M = np.random.normal(scale=0.01, size=(n_gate_neurons, n_gate_neurons)) # Anti-Hebbian weights within gate layer
        np.fill_diagonal(self.M, 0.)
        self.y = None # a vector, each element is the activity of one gate neuron
        self.Y = np.ones((n_gate_neurons, ))*100
        self.W_hg = np.zeros((n_input_neurons, n_gate_neurons)) # Weights from gate layer to hidden layer
        self.active_y_idx = 0 # the index of the gate neuron with the largest activity

    def get_gate_layer_output(self, x, state_changeQ=True, n_iterations=10) -> None:
        """
        Calculate self.y, self.active_y_idx. If state_changeQ is True, self.Y will also be changed. 
        n_iterations: not used.
        """
        w_mul_x = self.W@x
        # sigmoid = lambda x: 1/(1. + np.exp(-x))
        # map_fun = lambda y: sigmoid(w_mul_x - self.M@y)
        # y = np.copy(w_mul_x)
        # for _ in range(n_iterations):
        #     y = map_fun(y)
        def fun(x2):
            return x2 - np.maximum(w_mul_x-self.M@x2, 0)
            # return x2 - sigmoid(w_mul_x-self.M@x2)
        y = optimize.fsolve(fun, w_mul_x)
        self.y = y

        if state_changeQ is True:
            self.Y = self.decay1*self.Y + y**2
        self.active_y_idx = np.argmax(y)

    def get_hidden_layer_output(self, x) -> np.ndarray:
        """Calculate the hidden layer pattern by XOR operation"""
        tmp = x + self.W_hg[:, self.active_y_idx]
        return np.heaviside(tmp-0.5, 0) - np.heaviside(tmp-1.5, 0)
    
    def reconstruct_input(self, hidden) -> np.ndarray:
        """
        Reconstruct the input pattern based on the hidden layer pattern (assuming the self.active_y_idx has the correct value). 
        Calculating hidden layer pattern from x and calculating x from the hidden layer pattern share exactly the same procedure."""
        return self.get_hidden_layer_output(hidden)
        
    def update_weights(self, x) -> None:
        """Update self.W, self.M, self.W_hg"""
        tmp1, tmp2 = (self.y/self.Y).reshape((-1, 1)), (self.y**2/self.Y).reshape((-1, 1))
        self.W += tmp1@(x.reshape((1, -1))) - tmp2*self.W
        self.M += tmp1@(self.y.reshape((1, -1))) - tmp2*self.M
        np.fill_diagonal(self.M, 0.)
        self.W_hg[:, self.active_y_idx] = self.decay2*self.W_hg[:, self.active_y_idx] + x*(1-self.decay2)

class Hopfields():
    def __init__(self, n_neurons, n_clusters, decay=0.99) -> None:
        """
        This class contatins n_clusters individual sparse hopfield networks. 
        """
        self.n_neurons = n_neurons
        self.decay = decay
        self.W = np.zeros((n_clusters, self.n_neurons, self.n_neurons))

    def weight_update(self, x, active_clutser_idx) -> None:
        f = np.sum(x)/self.n_neurons # coding level
        self.W[active_clutser_idx] = self.decay*self.W[active_clutser_idx] + (np.einsum('i,j->ij', x-f,  x-f))/self.n_neurons
        np.fill_diagonal(self.W[active_clutser_idx], 0.) # diagonal elements should be 0

    def retrieve(self, x, active_cluster_idx, n_iterations=10) -> np.ndarray:
        for _ in range(n_iterations):
            f = np.sum(x)/self.n_neurons
            x = ((self.W[active_cluster_idx]@x) - f/2 > 0).astype(float)
        return x

class TreeNetwork():
    def __init__(self, n_input_neurons, tree_struct_list, tree_depth=0, tree_decay1_list=(0.99, ), tree_decay2_list=(0.99, ), decay3=0.99) -> None:
        """
        A tree-like network that does hierarchical memory classification storage and retrieval. 
        tree_struct_list: a tuple like (n1, n2, n3, ...). n1 is the number of descendents, n2 is the number of 1st-generation descendants, etc. This tree-like network will be constructed recursively according to this tuple. 
        tree_depth: the depth of this tree.
        tree_decay1_list: used to decay Y in SNMF module, it has the same structure as tree_struct_list; 
        tree_decay2_list: used to decay W_hg in SNMF module, it has the same structure as tree_struct_list;
        decay3: used to decay the Hopfield weights"""

        self.tree_depth = tree_depth
        self.n_gate_units = tree_struct_list[0] 
        decay1 = tree_decay1_list[0]
        decay2 = tree_decay2_list[0]

        self.snmf = SNMF(n_input_neurons, self.n_gate_units, decay1, decay2)
        self.next_tree_stuct_list = tree_struct_list[1:]
        self.hidden = np.zeros(n_input_neurons)
        if len(tree_struct_list) > 1:
            self.sub_trees = [TreeNetwork(n_input_neurons, self.next_tree_stuct_list, tree_depth=self.tree_depth+1, tree_decay1_list=tree_decay1_list[1:], tree_decay2_list=tree_decay2_list[1:], decay3=decay3) for _ in range(self.n_gate_units)]
        else: 
            self.sub_trees = []
            self.hopfields = Hopfields(n_input_neurons, n_clusters=self.n_gate_units, decay=decay3)
        self.count_record = np.zeros(self.n_gate_units) # The i-th element is the number of inputs that are classified as the i-th cluster so far. It is used to estimate the classification error. 

    def __repr__(self) -> str:
        rep = f"depth of the current layer: {self.tree_depth}, number of gate units: {self.n_gate_units}.\n"
        if self.sub_trees == []:
            rep = rep + f"This is the last classification layer, the next layer consists of {self.n_gate_units} hopfield networks\n"
        else:
            rep = rep + f"Each gate unit in this layer controls a tree of gate layers that has tree struct {self.next_tree_stuct_list}.\n"
        return rep

    def update_weights(self, x, trainSNMFQ=True, trainHopfieldQ=True) -> None:
        """trainSNMFQ: whether to change the weights in SNMF modules;
        trainHopfieldQ: whether to change the Hopfield weights. """
        if trainSNMFQ is True:
            self.snmf.get_gate_layer_output(x, state_changeQ=True)
            self.snmf.update_weights(x)
        else: 
            self.snmf.get_gate_layer_output(x, state_changeQ=False)
        y = self.snmf.get_hidden_layer_output(x) # input for the next layer
        if self.sub_trees == []:
            if trainHopfieldQ is True:
                self.hopfields.weight_update(y, self.snmf.active_y_idx)
        else:
            self.sub_trees[self.snmf.active_y_idx].update_weights(y, trainSNMFQ, trainHopfieldQ)
        return

    def forward_classify(self, x, state_changeQ=False, countingQ=False, n_hopfield_iterations=0):
        """Pass x forward, do classification. At the final layer, use hopfield network to do retrieval.
        To disable hopfield retrieval, set n_hopfield_iterations = 0
        state_changeQ: whether to change self.Y in the SNMF module of the current layer.
        countingQ: whether to update self.count_record"""
        self.snmf.get_gate_layer_output(x, state_changeQ=state_changeQ)
        self.hidden = self.snmf.get_hidden_layer_output(x) # input for the next layer
        active_gate_unit_idx = self.snmf.active_y_idx
        if self.sub_trees == []:
            classification_result = [active_gate_unit_idx]
            residual = self.hidden
            residual = self.hopfields.retrieve(residual, active_cluster_idx=active_gate_unit_idx, n_iterations=n_hopfield_iterations)
        else:
            tmp = self.sub_trees[active_gate_unit_idx].forward_classify(self.hidden, state_changeQ, countingQ, n_hopfield_iterations)
            classification_result = [active_gate_unit_idx] + tmp[0]
            residual = tmp[1]
        
        if countingQ is True:
            self.count_record[active_gate_unit_idx] += 1

        return (classification_result, residual)

    def backward_reconstruction(self, x) -> np.ndarray:
        """recursive backward reconstruction, Hopfield network is not involved."""
        if self.sub_trees != []:
            x = self.sub_trees[self.snmf.active_y_idx].backward_reconstruction(x)
        return self.snmf.reconstruct_input(x)

    def retrieve(self, x, n_hopfield_iterations, countingQ=False) -> np.ndarray:
        """x is the memory to be retrieved (usually with noise)"""
        _, residual  = self.forward_classify(x, state_changeQ=False, countingQ=countingQ, n_hopfield_iterations=n_hopfield_iterations)
        return self.backward_reconstruction(residual)

    def initialize(self, all_descendents, n_iterations=1) -> None:
        """
        Use n_iteration*n_clusters patterns to initialize the weights in SNMF module. 
        all_descendents is also a tree whose leaves are patterns to be stored. len(all_descendents) should match self.n_gate_units"""
        tmp = jnp.array([all_descendents[(i, )+(0, )*(len(all_descendents.shape)-2)] for i in range(self.n_gate_units)])
        for _ in range(n_iterations):
            for idx in range(self.n_gate_units):
                x = tmp[idx]
                y = np.zeros((self.n_gate_units, ))
                y[idx] = 1.
                self.snmf.active_y_idx = idx
                self.snmf.y = y
                self.snmf.update_weights(x)
        if self.sub_trees == []:
            return 
        else: 
            for idx in range(self.n_gate_units):
                self.sub_trees[idx].initialize(all_descendents[idx], n_iterations)

    def get_misclassification_percentage(self) -> float:
        avg = np.mean(self.count_record)
        return np.sum(np.abs(self.count_record-avg))/2/np.sum(self.count_record)
    
    def reset_counting_record(self) -> None:
        self.count_record = np.zeros(self.n_gate_units)
        if self.sub_trees != []:
            for idx in range(self.n_gate_units):
                self.sub_trees[idx].reset_counting_record()

    def reset_hopfield_weights(self) -> None:
        if self.sub_trees != []:
            for idx in range(self.n_gate_units):
                self.sub_trees[idx].reset_hopfield_weights()
        else:
            self.hopfields.W = np.zeros(self.hopfields.W.shape)

class UltraMetricTree():
    def __init__(self, key, n_neurons, tree_struct_list, tree_diff_ratio_list):
        """tree_struct_list = (n1, n2, ....) means that the root ancestor has n1 1st-generation decendents, and each of them has n2 2nd-generation descendents, ..."""
        self.key, subkey = jax.random.split(key)
        self.tree_struct_list = tree_struct_list # 1-dim list
        self.tree_diff_ratio_list = tree_diff_ratio_list # 1-dim list
        self.n_neurons = n_neurons
        self.root_ancestor = jnp.zeros((n_neurons, ), dtype=int)
        self.descendents = [self.root_ancestor] # a tree (nested list), each leaf is a pattern
        self.keys = [subkey] # a tree (nested list), each leaf is a key

    def get_direct_descendents(self, ancestor, key, n_patterns_per_cluster, diff_ratio=0.1):
        n_neurons = len(ancestor)

        # approx diff_ratio
        # diffs = jax.random.choice(key, jnp.array([0, 1]), (n_patterns_per_cluster, n_neurons), p=jnp.array([1-diff_ratio, diff_ratio]))

        #exact_diff_ratio
        n_diff = int(diff_ratio*n_neurons)
        tmp = jnp.concatenate((jnp.ones(n_diff, dtype=int), jnp.zeros(n_neurons-n_diff, dtype=int)))
        diffs = jnp.array([tmp for _ in range(n_patterns_per_cluster)])
        diffs = jax.random.permutation(key, diffs, axis=-1, independent=True)

        return jnp.bitwise_xor(diffs, ancestor)

    def get_next_tree(self, pattern_tree, key_tree, n_patterns_per_cluster, diff_ratio):
        def fun(key, ancestor):
            descendents = self.get_direct_descendents(ancestor, key, n_patterns_per_cluster, diff_ratio)
            return [descendents[i] for i in range(len(descendents))]
        self.descendents = jax.tree_map(fun, self.keys, self.descendents)
        self.keys = jax.tree_map(lambda x: list(jax.random.split(x, n_patterns_per_cluster)), self.keys)

    def construct_tree(self):
        for i in range(len(self.tree_struct_list)):
            n_patterns_per_cluster = self.tree_struct_list[i]
            diff_ratio = self.tree_diff_ratio_list[i]
            self.get_next_tree(self.descendents, self.keys, n_patterns_per_cluster, diff_ratio)
        
        