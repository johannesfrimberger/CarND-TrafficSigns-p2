"""
You need to change the Add() class below.
"""

class Neuron:
    def __init__(self, inbound_neurons=[]):
        # Neurons from which this Node receives values
        self.inbound_neurons = inbound_neurons
        # Neurons to which this Node passes values
        self.outbound_neurons = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_neurons` and
        store the result in self.value.
        """
        self.value = sum([el.value for el in self.inbound_neurons])

    def backward(self):
        """
        Backward propagation.

        Compute the gradient of the current node with respect
        to the input neurons. The gradient of the loss with respect
        to the current neuron should already be computed in the `gradients`
        attribute of the output neurons.
        """
        raise NotImplemented


class Input(Neuron):
    def __init__(self):
        # an Input neuron has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Neuron.__init__(self)

    # NOTE: Input node is the only node where the value
    # is passed as an argument to forward().
    #
    # All other neuron implementations should get the value
    # of the previous neurons from self.inbound_neurons
    #
    # Example:
    # val0 = self.inbound_neurons[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value:
            self.value = value


class Add(Neuron):
    def __init__(self, x, y):
        Neuron.__init__(self, [x, y])

    def forward(self):
        """
        Set the value of this neuron to the sum of it's inbound_nodes.

        Your code here!
        """
        self.value = sum([el.value for el in self.inbound_neurons])


"""
No need to change anything below here!
"""


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_neuron`: A neuron in the graph, should be the output neuron (have no outgoing edges).
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_neuron.value

"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

x, y = Input(), Input()

f = Add(x, y)

f2 = Add(f, y)

feed_dict = {x: 10, y: 5}

sorted_neurons = topological_sort(feed_dict)
output = forward_pass(f, sorted_neurons)
output2 = forward_pass(f2, sorted_neurons)
print("({} + {}) + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[y], output2))