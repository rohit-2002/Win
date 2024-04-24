#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

const int MAX_NODES = 100000;

// Node structure for the tree
struct TreeNode {
    int data;
    TreeNode *left, *right;
};

// Class for handling tree operations
class Tree {
public:
    TreeNode* buildTree(int num_elements);
    void bfs(TreeNode* root);
private:
    TreeNode* insertNode(TreeNode* root, int data);
};

TreeNode* Tree::insertNode(TreeNode* root, int data) {
    if (!root) {
        root = new TreeNode; 
        root->left = NULL; 
        root->right = NULL; 
        root->data = data; 
    } else {
        queue<TreeNode *> q; 
        q.push(root);
        while (!q.empty()) {
            TreeNode *temp = q.front();
            q.pop();
            if (temp->left == NULL) {
                temp->left = new TreeNode; 
                temp->left->left = NULL; 
                temp->left->right = NULL; 
                temp->left->data = data;
                break;
            } else {
                q.push(temp->left);
            }
            if (temp->right == NULL) {
                temp->right = new TreeNode; 
                temp->right->left = NULL; 
                temp->right->right = NULL; 
                temp->right->data = data; 
                break;
            } else {
                q.push(temp->right);
            }
        }
    }
    return root;
}

TreeNode* Tree::buildTree(int num_elements) {
    TreeNode *root = NULL; 
    
    cout << "Enter data for " << num_elements << " elements:\n";
    for (int i = 0; i < num_elements; ++i) {
        int data;
        cout << "Element " << i+1 << ": ";
        cin >> data;
        root = insertNode(root, data);
    }
    
    return root;
}

void Tree::bfs(TreeNode* root) {
    if (!root) return;

    queue<TreeNode*> q; 
    q.push(root);
    while (!q.empty()) {
        int qSize = q.size(); 
        for (int i = 0; i < qSize; i++) {
            TreeNode* currNode = q.front();
            q.pop();
            cout << "\t" << currNode->data;
            
            if (currNode->left)
                q.push(currNode->left); 
            if (currNode->right)
                q.push(currNode->right);
        }
    }
}

// Class for handling graph operations
class Graph {
public:
    void dfs(int startNode, int numNodes, vector<vector<int>>& graph);
private:
    void dfsUtil(int currentNode, vector<vector<int>>& graph, vector<bool>& visited);
};

void Graph::dfsUtil(int currentNode, vector<vector<int>>& graph, vector<bool>& visited) {
    stack<int> stack;
    stack.push(currentNode);

    while (!stack.empty()) {
        int node = stack.top();
        stack.pop();

        if (!visited[node]) {
            visited[node] = true;
            cout << node << " ";

            for (int adjacentNode : graph[node]) {
                if (adjacentNode >= 0 && adjacentNode < graph.size() && !visited[adjacentNode]) {
                    stack.push(adjacentNode);
                }
            }
        }
    }
}

void Graph::dfs(int startNode, int numNodes, vector<vector<int>>& graph) {
    vector<bool> visited(numNodes, false);
    dfsUtil(startNode, graph, visited);
}

int main() {
    Tree tree;
    Graph graph;

    int num_elements;
    cout << "Enter the number of elements you want to insert: ";
    cin >> num_elements;
    
    if (num_elements <= 0) {
        cout << "Invalid number of elements." << endl;
        return 1;
    }
    
    TreeNode* root = tree.buildTree(num_elements);

    cout << "Breadth-first traversal: ";
    tree.bfs(root);
    cout << endl;

    int numNodes, numEdges, startNode;
    cout << "\nEnter the number of nodes: ";
    cin >> numNodes;
    cout << "Enter the number of edges: ";
    cin >> numEdges;
    cout << "Enter the starting node of the graph: ";
    cin >> startNode;

    vector<vector<int>> graphData(numNodes);
    
    cout << "Enter pairs of nodes for the edges:\n";
    for (int i = 0; i < numEdges; i++) {
        int u, v;
        cout << "Edge " << i + 1 << ": ";
        cin >> u >> v;
        graphData[u].push_back(v);
        graphData[v].push_back(u);
    }

    cout << "Depth-first traversal starting from node " << startNode << ": ";
    graph.dfs(startNode, numNodes, graphData);
    cout << endl;

    return 0;
}
