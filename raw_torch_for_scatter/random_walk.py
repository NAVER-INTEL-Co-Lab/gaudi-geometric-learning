import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


class Node2Vec(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(Node2Vec, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, node_pairs):
        src, dst = node_pairs[:, 0], node_pairs[:, 1]
        return torch.sum(self.embeddings(src) * self.embeddings(dst), dim=1)


def random_walk(graph, start_node, walk_length, p, q):
    walk = [start_node]
    current_node = start_node

    for _ in range(walk_length):
        neighbors = graph[current_node]
        if not neighbors:
            break

        # Transition probabilities
        if len(walk) > 1:
            prev_node = walk[-2]
            weights = []
            for neighbor in neighbors:
                if neighbor == prev_node:
                    weights.append(1 / p)  # backtrack
                elif neighbor in graph[prev_node]:
                    weights.append(1)  # return to a neighbor
                else:
                    weights.append(1 / q)  # explore new neighbors
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()

            current_node = np.random.choice(neighbors, p=weights)
        else:
            current_node = random.choice(neighbors)

        walk.append(current_node)

    return walk


def generate_walks(graph, num_walks, walk_length, p, q):
    walks = []
    for node in graph.keys():
        for _ in range(num_walks):
            walk = random_walk(graph, node, walk_length, p, q)
            walks.append(walk)
    return walks


def create_node_pairs(walks):
    pairs = []
    for walk in walks:
        for i, node in enumerate(walk):
            # Create pairs with context nodes
            for j in range(max(0, i - 2), min(len(walk), i + 3)):
                if j != i:
                    pairs.append((node, walk[j]))
    return pairs


def train_node2vec(graph, num_nodes, embedding_dim, num_walks, walk_length, p, q, epochs=100, lr=0.01):
    model = Node2Vec(num_nodes, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss()

    walks = generate_walks(graph, num_walks, walk_length, p, q)
    node_pairs = create_node_pairs(walks)

    for epoch in range(epochs):
        total_loss = 0
        for src, dst in node_pairs:
            optimizer.zero_grad()
            src_embedding = model.embeddings(torch.tensor(src))
            dst_embedding = model.embeddings(torch.tensor(dst))
            score = (src_embedding * dst_embedding).sum()
            loss = loss_function(score.unsqueeze(0), torch.tensor([1.0]))  # Positive pairs
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

    return model.embeddings.weight.data


# Example graph as adjacency list
graph = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 4],
    3: [1, 4, 5],
    4: [2, 3, 5],
    5: [3, 4]
}

# Parameters
num_nodes = len(graph)
embedding_dim = 64
num_walks = 10
walk_length = 5
p = 1
q = 1

# Train Node2Vec
embeddings = train_node2vec(graph, num_nodes, embedding_dim, num_walks, walk_length, p, q)

# Display node embeddings
print("Node Embeddings:\n", embeddings)