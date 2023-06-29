# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "370bac4919b358e3a43a3ccb70a587f3", "grade": false, "grade_id": "cell-e12df818732a12f6", "locked": true, "schema_version": 3, "solution": false, "task": false}
#
# # Exercise Sheet No. 10
#
# ---
#
# > Machine Learning for Natural Sciences, Summer 2023, Jun.-Prof. Pascal Friederich, pascal.friederich@kit.edu
# > 
# > Deadline: July 11th 2023, 8:00 am
# >
# > Container version 1.0.1
# >
# > Tutor: patrick.reiser@kit.edu
# >
# > **Please ask questions in the forum/discussion board and only contact the Tutor when there are issues with the grading**
# ---
#
# ---
#
# **Topic**: This exercise sheet will introduce you to machine learning on graphs.

# %% [markdown]
# Please add here your group members' names and student IDs. 
#
# Names: Robin Maurer, Francisca Azocar Dannemann, Marcus Fledler
#
# IDs: 2462304, 2480646, 2494460

# %% [markdown]
# # Graph Theory
# From [wikipedia](https://en.wikipedia.org/wiki/Graph_theory): "In mathematics, graph theory is the study of graphs, which are mathematical structures used to model pairwise relations between objects. A graph in this context is made up of vertices (also called nodes or points) which are connected by edges (also called links or lines). A distinction is made between undirected graphs, where edges link two vertices symmetrically, and directed graphs, where edges link two vertices asymmetrically. Graphs are one of the principal objects of study in discrete mathematics."
#
# In one restricted but very common sense of the term, a graph is an ordered pair $G = ( V , E )$ comprising:
#
# * The vertex set $V$ of vertices (also called nodes or points);
# * The edge set $E\subseteq \{ \{x, y\} \mid x, y \in V \}$ edges (also called links or lines), which are unordered pairs of vertices (that is, an edge is associated with two distinct vertices).
#
# To avoid ambiguity, this type of object may be called precisely an undirected simple graph. 

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7bfa34b19454d2a8084071e52f2d018a", "grade": false, "grade_id": "cell-0268a3c2d823ea3c", "locked": true, "schema_version": 3, "solution": false, "task": false}
import matplotlib.pyplot as plt
import networkx as nx

# Example of a graph
G = nx.krackhardt_kite_graph()
nx.draw(G, pos=nx.kamada_kawai_layout(G))

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0205e19c86514d093200d1c0018cd049", "grade": false, "grade_id": "cell-ae513ca0cdb0800f", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Graph convolutional neural networks (GCN)
#
# Graph convolutional neural networks are a natural extension of CNNs for graph-structured data. A simple but efficient Graph Neural Network was introduced in ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907) by Kipf et al. (2016). Below is a description of the model and its applications taken are from https://tkipf.github.io/graph-convolutional-networks/. We will implement this model and test it on some graph data.
#
# ### GCNs Part I: Definitions
#
# Currently, most graph neural network models have a somewhat universal architecture in common. We will refer to these models as Graph Convolutional Networks (GCNs); convolutional, because filter parameters are typically shared over all locations in the graph (or a subset thereof as in [Duvenaud et al., NIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html)).
#
# For these models, the goal is then to learn a function of signals/features on a graph $G = (V, E)$ which takes as input:
#
# * A feature description $x_i$ for every node $i$; summarized in a $N\times D$ feature matrix $X$ ($N$: number of nodes, $D$: number of input features)
# * A representative description of the graph structure in matrix form; typically in the form of an [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) $A$ (or some function thereof)
#
# and produces a node-level output $Z$ (an $N\times F$ feature matrix, where $F$ is the number of output features per node). Graph-level outputs can be modeled by introducing some form of pooling operation (see, e.g. [Duvenaud et al., NIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html)).
#
# Every neural network layer can then be written as a non-linear function
#
# $$H^{(l+1)}=f(H^{(l)}, A),$$
#
# with $H^{(0)}=X$ and $H^{(L)}=Z$ (or $z$ for graph-level outputs), $L$ being the number of layers. The specific models then differ only in how $f(⋅,⋅)$ is chosen and parameterized.
#
# ### GCNs Part II: A simple example
#
# As an example, let's consider the following very simple form of a layer-wise propagation rule:
#
# $$f(H^{(l)},A)= \sigma (AH^{(l)}W^{(l)}),$$
#
# where $W^{(l)}$ is a weight matrix for the $l$-th neural network layer and $\sigma(⋅)$ is a non-linear activation function like the ReLU. Despite its simplicity this model is already quite powerful (we'll come to that in a moment).
#
# But first, let us address two limitations of this simple model: multiplication with $A$ means that, for every node, we sum up all the feature vectors of all neighboring nodes but not the node itself (unless there are self-loops in the graph). We can "fix" this by enforcing self-loops in the graph: we simply add the identity matrix to $A$.
#
# The second major limitation is that $A$ is typically not normalized and therefore the multiplication with $A$ will completely change the scale of the feature vectors proportional to a node's degree. Normalizing $A$ such that all rows sum to one, i.e. $D^{−1}A$, where $D$ is the diagonal node degree matrix, gets rid of this problem. Multiplying with $D^{−1}A$ now corresponds to taking the average of neighboring node features. In practice, dynamics get more interesting when we use a symmetric normalization, i.e. $D^{−\frac{1}{2}} A D^{−\frac{1}{2}}$ (as this no longer amounts to mere averaging of neighboring nodes). Combining these two tricks, we essentially arrive at the propagation rule introduced in [Kipf & Welling](https://arxiv.org/abs/1609.02907) (ICLR 2017):
#
# $$f(H^{(l)},A)=\sigma \, ( \hat{D}^{−\frac{1}{2}} \hat{A} \hat{D}^{−\frac{1}{2}} H^{(l)}W^{(l)}),$$
#
# with $\hat{A}=A+I$, where $I$ is the identity matrix and $\hat{D}$ is the diagonal node degree matrix of $\hat{A}$.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6ac240ba1b5da3cf03da509472d4413c", "grade": false, "grade_id": "cell-9691d7e3b67b0bbe", "locked": true, "schema_version": 3, "solution": false, "task": false}
# For a better understanding, let us additionally build the connection between Message Passing Networks and Graph (Convolutional) Networks.
# In the message passing framework, for each node $v$, on each iteration $t$ we construct messages
# $$m_{v}^{t+1} = \sum_{w \in N(v)} M_{t}(h_{v}^{t},h_{w}^{t},e_{vw})$$
# and updates of a node's hidden state
# $$h_{v}^{t+1} = U(h_{v}^{t}, m_{v}^{t+1})$$
#
# with $h_{v}^{0}$ being the initial features of node $v$.
# While the matrix formulation may seem unrelated to the message passing framework at first sight, it is in fact one particular instance of message passing.
#
# Let $G = (V, E)$ be a graph and $n = |V|$. Let $A \in \mathbb{R}^{n \times n}$ be the graph's adjacency matrix induced by $E$. Let $X \in \mathbb{R}^{n \times D}$ be the node feature tensor. We use the slice notation as in Python, numpy, TensorFlow or PyTorch, i.e., $A_{0:}$ indexes the first row of $A$ which indicates
# whether node $0$ is connected to nodes $0 .. n$.
#
# The dot product $A_{0:}X$ then effectively performs two steps. First, during the multiplication step of the dot product, the $0-1$ entries of the row act as a (convolutional) filter on the node feature tensor $X$ by pointwise products, i.e., they define $N(v)$. Second, during the addition step of the dot product, $\sum_{w \in N(v)}$ is calculated. Since we do not perform any additional steps during message construction, our message function is simply $M_{t}(h_{v}^{t},h_{w}^{t},e_{vw}) = h_{w}^{t}.$
#
# Similarly, as we do not have any further operations to perform via matrix multiplication, our update function is thus $h_{v}^{t+1} = U(h_{v}^{t}, m_{v}^{t+1})= m_{v}^{t+1}$.
#
# Therefore, $AX$ effectively performs a full graph convolution step via matrix multiplication. As a result, the modification (like normalizing it) of $A$ results in a different filter and in turn yields a different node and graph representation. This vectorized implementation is especially efficient for dense graphs. For large sparse graphs, sparse matrix multiplication or edge lists are preferred due to the large number of multiplications by 0.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b9670fd2ed82b81e478d1065592ee18d", "grade": false, "grade_id": "cell-f648a16e58e30317", "locked": true, "schema_version": 3, "solution": false, "task": false}
#
# Next, we will take a closer look at how this type of model operates on a very simple example graph: Zachary's karate club network (make sure to check out the [Wikipedia article](https://en.wikipedia.org/wiki/Zachary%27s_karate_club)!). A social network of a karate club was studied by Wayne W. Zachary for a period of three years from 1970 to 1972. The network captures 34 members of a karate club, documenting links between pairs of members who interacted outside the club. During the study a conflict arose between the administrator "John A" and instructor "Mr. Hi" (pseudonyms), which led to the split of the club into two. Half of the members formed a new club around Mr. Hi; members from the other part found a new instructor or gave up karate. Based on collected data Zachary correctly assigned all but one member of the club to the groups they actually joined after the split.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7e3d6e46fb375ecb525f42911ce13864", "grade": false, "grade_id": "cell-3fa3bc6db7884ffa", "locked": true, "schema_version": 3, "solution": false, "task": false}
G = nx.karate_club_graph()
labels = [G.nodes()[i]["club"] for i in range(nx.number_of_nodes(G))]
print("Example of labels[5:15] :", labels[5:15])
labels_onehot = [0 if x == "Mr. Hi" else 1 for x in labels]
labels_color = ["green" if x == "Mr. Hi" else "red" for x in labels]
nx.draw_circular(G, with_labels=True, node_color=labels_color)
plt.show()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "45868681d02f97f94780fa5fcfbc0d58", "grade": false, "grade_id": "cell-ba3d1f3df6681688", "locked": true, "schema_version": 3, "solution": false, "task": false}
import scipy
import scipy.sparse as sp
import numpy as np

# Adjacency matrix as sparse matrix
A = nx.adjacency_matrix(G)
print(A.toarray())


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b3b73aeb658ce7a53e9edc94b3f8c39c", "grade": false, "grade_id": "cell-c447e57e44fd3c72", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### 10.1 Normalized adjacency matrix
# Compute the normalized adjacency matrix $A_{norm} = \hat{D}^{−\frac{1}{2}} \hat{A} \hat{D}^{−\frac{1}{2}}$, with $\hat{A}=A+I$, where $I$ is the identity matrix and $\hat{D}$ is the diagonal node degree matrix of $\hat{A}$.
# You have to implement ``compute_normalized_adj`` using scipy's sparse matrix functions, such as `.sum`, `sp.diag`, `sp.eye`, `.shape`, `.transpose` and `.dot`. It is okay to compute the power of the row- and column sum with `numpy`. You may have to check for `np.inf` here when taking $\hat{D}^{−\frac{1}{2}}$.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "5f7d48d0fb0ca2b2ba1bfb39b38491af", "grade": false, "grade_id": "task_compute_scaled_DAD", "locked": false, "schema_version": 3, "solution": true, "task": false}
def compute_normalized_adj(adj):
    """Compute scaled adjacency matrix D^-0.5*(A + I)*D^-0.5
    with the degree matrix D of (A+I).
    
    Args:
        adj (scipy.sparse): Sparse matrix representation of A with shape (N, N)
    
    Returns:
        sp.sparse.coo_matrix: Sparse matrix representation of D^-0.5*(A + I)*D^-0.5
    """
    # YOUR CODE HERE
    N = adj.shape[0]
    A_dach = adj + sp.identity(N)
    d = A_dach.sum(axis = 1)
    d_1 = np.power(d, -0.5)
    D_1 = sp.diags(d_1)
    a_norm = D_1.dot(A_dach.dot(D_1))
    #raise NotImplementedError()
    return a_norm.tocoo()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4b314882879cd20f8cd0d877e0a74b6d", "grade": false, "grade_id": "cell-8b18321511d58e0d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# test on the Karate Club network
A_norm = compute_normalized_adj(A)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f879a7887aa55ee8cef0b149eedda945", "grade": true, "grade_id": "test_A_scaled", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false}
assert isinstance(A_norm, scipy.sparse.coo_matrix)

test_arr_1 = np.array([
    [0.023256, 0.111369, 0.130766, 0.104957, 0.152499],
    [0.111369, 0.033333, 0.187867, 0.125656, 0.000000],
    [0.130766, 0.187867, 0.029412, 0.118033, 0.000000],
    [0.104957, 0.125656, 0.118033, 0.052632, 0.000000],
    [0.152499, 0.000000, 0.000000, 0.000000, 0.111111]
])

assert np.abs(A_norm.tocoo().toarray()[:5, :5] - test_arr_1).max() < 1e-4, "absolute difference too large"


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9aa010c7ec821652b33888b876fa6782", "grade": false, "grade_id": "cell-51d74fb60a8c4273", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now we implement a TensorFlow model, which accepts the node features and the normalized adjacency matrix as inputs. The actual convolution realized by a (sparse) matrix multiplication is given in the layer below: `GCNConvolution`. This means the scaling does not to be performed within the model but can be done beforehand via `compute_normalized_adj`.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b90b9a77dfe1f03a88b9d41c80e6fbaf", "grade": false, "grade_id": "cell-6761c99078bb6033", "locked": true, "schema_version": 3, "solution": false, "task": false}
import tensorflow as tf


class GCNConvolution(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(GCNConvolution, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        adj, x = inputs
        return tf.sparse.sparse_dense_matmul(adj, x)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c75b65cb4426831b45b905697d20b877", "grade": false, "grade_id": "cell-42be802a92429c0a", "locked": true, "schema_version": 3, "solution": false, "task": false}
# We set up the model in the functional API for TensorFlow Keras. You can read about this here: https://www.tensorflow.org/guide/keras/functional. A skeleton of the intended model is shown below. You have to add 3 layers of convolution. For the weights $W^{(l)}$ you can simply use `tf.keras.layers.Dense` with `linear` activation and without any `use_bias=False` and for the non-linearity after the matrix multiplication you can use `tf.keras.layers.Activation`. As activation, you can use ``"relu"`` and the dimension of the kernel is given already below as `hidden_dim`. We wrote some pseudo-code in the section, where you have to implement the model-part.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "7304ceff501aa07066e8b49a553a2cf3", "grade": false, "grade_id": "task_GCN_sparse_implement", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Model properties
hidden_dim = 34
target_dim = 2
depth = 3
# Model definition
input_feat = tf.keras.layers.Input(shape=(34,), name="node_input", dtype="float32")  # Node features
input_adj = tf.keras.layers.Input(shape=(34,), name="adj_input", dtype="float32", sparse=True)  # Scaled Adjacency matrix

x = input_feat
for i in range(depth):
    # Pseudo-Code of the model "x = D^-0.5*(A + I)*D^-0.5 * W *x"
    #
    # x = W*x (via tf.keras.layers.Dense(hidden_dim))
    # x = A_scaled * x (via GCNConvolution())
    # x = sigma(x) (via tf.keras.layers.Activation("relu"))
    #
    
    # YOUR CODE HERE
    dense = tf.keras.layers.Dense(hidden_dim, activation = "linear", use_bias = False)
    activate = tf.keras.layers.Activation("relu")
    gcnconv = GCNConvolution()
    x = dense(x)
    x = gcnconv((input_adj,x))
    x = activate(x)
    #raise NotImplementedError()

out_classes = tf.keras.layers.Dense(target_dim, activation="softmax")(x)
model = tf.keras.models.Model(inputs=[input_adj, input_feat], outputs=out_classes)
model.summary()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c73efa782ede41b2823b3c601eac3f30", "grade": false, "grade_id": "cell-6a94b5109e1b7e66", "locked": true, "schema_version": 3, "solution": false, "task": false}
# We will train the model above on a semi-supervised learning procedure, meaning that we will train the model on the Karate club network with a couple of nodes cloaked or covered and then test if the network can predict the right assignment of the unknown students. Since the network does not have features `X`, we will simply assume `X=I`. Check the code below.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "3ace1a882f46ff6db87aeb02e83867b3", "grade": false, "grade_id": "cell-0242d3b68ef1ea35", "locked": true, "schema_version": 3, "solution": false, "task": false}
from sklearn.model_selection import train_test_split

# Features and labels
X = np.eye(34)
Y = np.array([[1, 0] if x == "Mr. Hi" else [0, 1] for x in labels])
As = A_norm.tocsr().sorted_indices().tocoo()
# Train validation mask to cover nodes in the training
index_karate = np.arange(34)
ind_train, ind_val = train_test_split(index_karate, test_size=0.25, random_state=42)
val_mask = np.zeros_like(index_karate)
train_mask = np.zeros_like(index_karate)
val_mask[ind_val] = 1
train_mask[ind_train] = 1
#Draw the graphs
plt.figure(figsize=(15, 5))
plt.subplot(121)
nx.draw_circular(G, with_labels=True, node_color=[x if train_mask[i] == 1 else "black" for i, x in enumerate(labels_color)])
plt.title("Training Graph")
plt.subplot(122)
nx.draw_circular(G, with_labels=True, node_color=[x if val_mask[i] == 1 else "black" for i, x in enumerate(labels_color)])
plt.title("Validation Graph")
plt.show()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "dff324e69578f25f3b932cafd21d319e", "grade": false, "grade_id": "cell-da26be3c5c14fdca", "locked": true, "schema_version": 3, "solution": false, "task": false}
# We can use TensorFlow Keras API to train the model with some modifications. This is a somewhat hacky solution and not really as the Keras training API is intended. We directly insert the sparse matrix and Keras will think of the nodes as samples. We therefore have to fix the `batch_size=34`. Also we must prevent shuffling of the nodes (that will destroy our graph) and set `shuffle=False`. The covering of validation node labels can be realized with a `sample_weight=train_mask` that sets the validation nodes to zero in the loss (since nodes correspond to samples in this case).

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "6390fbfceae2b8f81321fd69b89e49b0", "grade": false, "grade_id": "cell-91c6de5429da3aac", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Compile model with optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    weighted_metrics=["categorical_accuracy"]
)

hist = model.fit(
    x=[As, X], y=Y,
    epochs=100,
    batch_size=34,
    verbose=1,
    shuffle=False,  # Since we do not really have batches, nodes must not be shuffled
    sample_weight=train_mask  # Important to hide values from
)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "18a7d3f2e35805c699fc22d8b9143812", "grade": false, "grade_id": "cell-f8d984ab33dd02a4", "locked": true, "schema_version": 3, "solution": false, "task": false}
val_loss = model.evaluate([As, X], Y, batch_size=34, sample_weight=val_mask)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "12b7caba6c7dbbb554445806321026ea", "grade": false, "grade_id": "cell-c1b393c7c659c9ad", "locked": true, "schema_version": 3, "solution": false, "task": false}
pred = model.predict([As, X], batch_size=34)
pred_val = pred[np.array(val_mask, dtype="bool")]
pred_train = pred[np.array(train_mask, dtype="bool")]
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.scatter(pred_train[:, 0], pred_train[:, 1], s=100, alpha=0.5,
            c=[x for i, x in enumerate(labels_color) if train_mask[i] == 1])
plt.plot((0, 1), c="black")
plt.title("Training Prediction")
plt.subplot(122)
plt.scatter(pred_val[:, 0], pred_val[:, 1], s=100, alpha=0.5, c=[x for i, x in enumerate(labels_color) if val_mask[i] == 1])
plt.title("Validation Prediction")
plt.plot((0, 1), c="black")
plt.show()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "af8871588bfe1e99c0828e25db14a56f", "grade": true, "grade_id": "test_GCN_sparse_acc", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false}
assert val_loss[-1] > 0.8
assert hist.history["categorical_accuracy"][-1] > 0.95

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5e3ce1d54cd65fe64886dd4737e8956f", "grade": false, "grade_id": "cell-c8523643ca5d8313", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Why is it not surprising that the training accuracy approaches 1.0 ? But why is it interesting that the validation accuracy is high?

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "99812680e397df93b3f20a3f0411ddf8", "grade": false, "grade_id": "task_answer_GCN", "locked": false, "schema_version": 3, "solution": true, "task": false}
#Kontrolle: antworten kontrollieren
answer_training_acc = "We only train the model on the Karate Club network, with some modifications, \
therefore there is overfitting, and it is not surprising that the accuracy approaches 1.0"
answer_validation_acc = "It is interesting that the validaiton accuracy is high, as this indicates that our model also \
accurately predicted the outcome for nodes that were unknown during training. This shows that this model works well \
and could yield good results when applied to other networks"

# YOUR CODE HERE
#raise NotImplementedError()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f317589a20401ebe0f3da83a19df6d52", "grade": true, "grade_id": "test_answer_GCN", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
assert isinstance(answer_training_acc, str)
assert isinstance(answer_validation_acc, str)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "82b0a939aa42d4dc46f240a6b509a78e", "grade": false, "grade_id": "cell-53cb7b55d6253a1e", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # 10.2 Molecules
# Molecules can also be represented as graphs. In this case we have many small(er) graphs and are interested in graph-embedding or graph classification rather than node embedding or classification. For this part we use the MUTAG dataset that classifies a set of small molecules as mutagenes.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "6e7f3b8592a55bd27e2e982bc8424156", "grade": false, "grade_id": "cell-281cbb016a682e34", "locked": true, "schema_version": 3, "solution": false, "task": false}
import requests
import zipfile
import os

data_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip"
if not os.path.exists("MUTAG.zip"):
    print("Downloading dataset ...")
    r = requests.get(data_url)
    _ = open("MUTAG.zip", "wb").write(r.content)
    print("Extracting dataset ...")
    archive = zipfile.ZipFile("MUTAG.zip", "r")
    archive.extractall()
    print("Extracting done.")


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "620a014919a63e142330cc395fa94892", "grade": false, "grade_id": "cell-d5eaad9d551c9823", "locked": true, "schema_version": 3, "solution": false, "task": false}
def read_mutag_text(filename):
    with open(filename, "r") as f:
        lines = f.read().split("\n")
    indices = [list(map(int, l.split(", "))) for l in lines if l]
    return np.array(indices, dtype="int")


# %%
# Read all mutag files
mutag_A = read_mutag_text("MUTAG/MUTAG_A.txt")
mutag_E = read_mutag_text("MUTAG/MUTAG_edge_labels.txt")
mutag_G = read_mutag_text("MUTAG/MUTAG_graph_indicator.txt")
mutag_N = read_mutag_text("MUTAG/MUTAG_node_labels.txt")
mutag_L = read_mutag_text("MUTAG/MUTAG_graph_labels.txt")

print("Shape of A:", mutag_A.shape)
print("Shape of Edges:", mutag_E.shape)
print("Shape of Graph-ID:", mutag_G.shape)
print("Shape of Nodes:", mutag_N.shape)
print("Shape of Graph-label:", mutag_L.shape)

# Want to start index from zero and not from 1
mutag_A -= 1
mutag_G -= 1

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "8ccd6822aa9a2050d40f9f69a9168098", "grade": false, "grade_id": "cell-565522fc577b76fb", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Separate the molecular graphs:
atom_code = ["C", "N", "O", "F", "I", "Cl", "Br"]
atom_color_map = {"C": "grey", "N": "green", "O": "blue", "F": "purple", "I": "yellow", "Cl": "pink", "Br": "orange"}
all_graphs = nx.Graph()
for i in range(len(mutag_N)):
    one_hot_atom_embedding = [0] * len(atom_code)  # There are 7 elements in the dataset
    one_hot_atom_embedding[mutag_N[i][0]] = 1
    str_atom_name = atom_code[mutag_N[i][0]]
    all_graphs.add_node(i,
                        features=one_hot_atom_embedding,
                        atom_name=str_atom_name)
for i in range(len(mutag_A)):
    one_hot_bond_embedding = [0] * 4  # There are 3 bond types in the dataset
    one_hot_bond_embedding[mutag_E[i][0]] = 1
    all_graphs.add_edge(mutag_A[i][0], mutag_A[i][1],
                        features=one_hot_bond_embedding)
graphs = []
nodes = np.arange(0, len(mutag_N), 1)
for g in range(0, np.amax(mutag_G)):
    graphs.append(all_graphs.subgraph(nodes[mutag_G[:, 0] == g]).to_directed())


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "a1f922af259e2d075bc3aeaacc5e7201", "grade": false, "grade_id": "cell-c06bedd8bd8c422e", "locked": true, "schema_version": 3, "solution": false, "task": false}
def plot_sub_graph(gri_plot):
    nx.draw(
        G=graphs[gri_plot],
        pos=nx.kamada_kawai_layout(graphs[gri_plot]),
        labels={n: (graphs[gri_plot].nodes()[n]["atom_name"]) for n in graphs[gri_plot].nodes()},
        node_color=[atom_color_map[graphs[gri_plot].nodes()[n]["atom_name"]] for n in graphs[gri_plot].nodes()]
    )


plt.figure(figsize=(15,12))
plt.subplot(231)
plot_sub_graph(2)
plt.subplot(232)
plot_sub_graph(75)
plt.subplot(233)
plot_sub_graph(23)
plt.subplot(234)
plot_sub_graph(101)
plt.subplot(235)
plot_sub_graph(21)
plt.subplot(236)
plot_sub_graph(123)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "44d550668968a54792dcc9b6a31f7cb8", "grade": false, "grade_id": "cell-b4400beb2d6a696e", "locked": true, "schema_version": 3, "solution": false, "task": false}
As = [nx.adjacency_matrix(graphs[i]) for i in range(len(graphs))]
Xs = [np.array([g.nodes()[n]["features"] for n in g.nodes()]) for g in graphs]

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5deb44ffaedc198d5e9aeec5e9c741ba", "grade": false, "grade_id": "cell-52bf36f26f293aa0", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now we scale the matrix and already cast it to a dense array
# If you did not solve previous part you can do it with numpy functions here.
As = [compute_normalized_adj(a).toarray() for a in As]

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "474af86c4c0d5bf5755032a30061a7cf", "grade": false, "grade_id": "cell-4c4212d700cbfb83", "locked": true, "schema_version": 3, "solution": false, "task": false}
# To put multiple graphs of different size in a single tensor, we will use padding here. That means we simply fill up the tensor with zeros. However, since zeros can cause non-zero output (e.g. bias), a mask to ignore these values has to be added to the model. Keras also has masking capabilities for example for RNNs/LSTM and Masking passing between layers, but we will do it manually "by hand" or explicitly here.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "2480bec61961203f17bf2f2694ef1f24", "grade": false, "grade_id": "task_padding_A_X", "locked": false, "schema_version": 3, "solution": true, "task": false}
A_padded = np.zeros((187, 28, 28))
A_mask = np.zeros((187, 28, 28), dtype="bool")

X_padded = np.zeros((187, 28, 7))
X_mask = np.zeros((187, 28, 1), dtype="bool")

for i, a in enumerate(As):
    # Fill A_padded, A_mask with correct values
    # YOUR CODE HERE
    n = a.shape[0]
    A_padded[i,:n,:n] = a
    A_mask[i,:n,:n] = np.ones((n,n))
    #raise NotImplementedError()

for i, x in enumerate(Xs):
    # Fill X_padded, X_mask with correct values
    # YOUR CODE HERE
    n = x.shape[0]
    X_padded[i,:n,:] = x
    X_mask[i,:n,:] = np.ones((n,1))
    #X_mask[i,:n,:] = np.ones((n,d))
    #raise NotImplementedError()

labels = mutag_L
labels[labels == -1] = 0  # labels are in {-1, 1}, we want them as {0, 1}

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "fb73979383403629c97829438e8e8103", "grade": true, "grade_id": "test_padding_AX", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false}
test_arr_3 = np.array([
    [0.33333333, 0.33333333, 0.00000000, 0.00000000, 0.00000000],
    [0.33333333, 0.33333333, 0.28867513, 0.00000000, 0.00000000],
    [0.00000000, 0.28867513, 0.25000000, 0.25000000, 0.00000000],
    [0.00000000, 0.00000000, 0.25000000, 0.25000000, 0.28867513],
    [0.00000000, 0.00000000, 0.00000000, 0.28867513, 0.33333333]
])

test_arr_4 = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 1.])

assert np.abs(A_padded[75, :5, :5] - test_arr_3).max() < 1e-4
assert np.abs(X_padded[75, :10, 0] - test_arr_4).max() < 1e-4

assert np.abs(A_padded[75, 10:, 10:]).max() < 1e-4
assert np.abs(X_padded[75, 10:, :]).max() < 1e-4


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "64339b19375529030f8cb94ad3ec9bb4", "grade": false, "grade_id": "cell-c32f913f98b651f8", "locked": true, "schema_version": 3, "solution": false, "task": false}
# The model can be defined similar to the previous task (skeleton below). However, this time you can do the matrix multiplication with `tf.keras.layers.Dot` and applying the mask with `tf.keras.layers.Multiply`. We will derive a graph-embedding by simply averaging all the node-embeddings of the last layer. For this purpose, we use the `Pooling` layer after the graph convolutions.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "9203a7ff9b2dc687af8693b99a44ad21", "grade": false, "grade_id": "cell-914f10c69414ea7c", "locked": true, "schema_version": 3, "solution": false, "task": false}
import tensorflow as tf


class Pooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Pooling, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.math.reduce_mean(inputs, axis=1)


# %% deletable=false nbgrader={"cell_type": "code", "checksum": "86d31cb0e43f5fd910686381f3d9a8bf", "grade": false, "grade_id": "task_GCN_padd_imp", "locked": false, "schema_version": 3, "solution": true, "task": false}
# Model properties
hidden_dim = 32
target_dim = 1
depth = 3

# Model definition
input_adj = tf.keras.layers.Input(shape=(28, 28), name="adj_input", dtype="float32")
input_x = tf.keras.layers.Input(shape=(28, 7), name="atom_input", dtype="float32")
input_x_mask = tf.keras.layers.Input(shape=(28, 1), name="atom_mask", dtype="float32")
x = input_x
for i in range(depth):
    # Pseudo-Code of the model "x = D^-0.5*(A + I)*D^-0.5 * W *x"
    #
    # x = W*x (via tf.keras.layers.Dense(hidden_dim))
    # x = Mask(x) (via tf.keras.layers.Multiply())
    # x = A_scaled * x (via tf.keras.layers.Dot() think about the "axes" argument)
    # x = sigma(x) (via tf.keras.layers.Activation("relu"))
    # x = Mask(x) (via tf.keras.layers.Multiply())
    #
    # YOUR CODE HERE
    dense = tf.keras.layers.Dense(hidden_dim, activation = "linear", use_bias = False)
    activate = tf.keras.layers.Activation("relu")
    x = dense(x)
    x = tf.keras.layers.Multiply()([x, input_x_mask])
    #Kontrolle axes bei layers dot ist 2,1 oder 1,2 oder ist es egal?
    x = tf.keras.layers.Dot(axes = (2,1) )([input_adj, x])
    x = activate(x)
    x = x = tf.keras.layers.Multiply()([x, input_x_mask])
    #raise NotImplementedError()

x_pool = Pooling()(x)
out_classes = tf.keras.layers.Dense(target_dim, activation="sigmoid")(x_pool)
model = tf.keras.models.Model(inputs=[input_adj, input_x, input_x_mask], outputs=out_classes)
model.summary()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "9311bc4ad5feca39f987f3fe029cd5ef", "grade": false, "grade_id": "cell-5f962097ce9e72af", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Compile model with optimizer and loss
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    weighted_metrics=["accuracy"]
)
hist = model.fit(
    x=[A_padded, X_padded, X_mask],
    y=labels,
    epochs=500,
    batch_size=64,
    verbose=1,
    shuffle=True,
    validation_split=0.15
)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "989ce5f776b5a71dc05cab81e37f3171", "grade": false, "grade_id": "cell-01c35b2f7f232988", "locked": true, "schema_version": 3, "solution": false, "task": false}
def plot_history(hist, validation_freq=1, scale=1):
    plt.figure()
    for key, loss in hist.history.items():
        np_loss = np.array(loss)
        if "val" in key:
            plt.plot(np.arange(np_loss.shape[0]) * validation_freq + validation_freq, np_loss, label=key)
        else:
            plt.plot(np.arange(np_loss.shape[0]), np_loss, label=key)

    plt.xlabel("Epochs")
    plt.ylabel("Loss ")
    plt.title("Loss vs. Epochs")
    plt.legend(loc="upper right", fontsize="x-small")
    plt.show()


plot_history(hist)

# %%
assert hist.history["accuracy"][-1] > 0.7
assert hist.history["val_accuracy"][-1] > 0.5

# %%
