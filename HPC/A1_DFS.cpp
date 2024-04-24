#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

using namespace std;

const int MAX_NODES = 100000;
vector<int> graph[MAX_NODES];
bool visited[MAX_NODES];

void depthFirstSearch(int startNode, int numNodes, int numEdges) {
    stack<int> stack;
    stack.push(startNode);
    
    while (!stack.empty()) {
        int currentNode = stack.top();
        stack.pop();
        
        if (!visited[currentNode]) {
            visited[currentNode] = true;
            cout << currentNode << " ";
            
            #pragma omp parallel for
            for (int i = 0; i < graph[currentNode].size(); i++) {
                int adjacentNode = graph[currentNode][i];
                if (!visited[adjacentNode]) {
                    stack.push(adjacentNode);
                }
            }
        }
    }
}

int main() {
    int numNodes, numEdges, startNode;
    
    cout << "Enter the number of nodes: ";
    cin >> numNodes;
    cout << "Enter the number of edges: ";
    cin >> numEdges;
    cout << "Enter the starting node of the graph: ";
    cin >> startNode;

    cout << "Enter pairs of nodes for the edges:\n";
    for (int i = 0; i < numEdges; i++) {
        int u, v;
        cout << "Edge " << i + 1 << ": ";
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    // Initialize visited array
    #pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
        visited[i] = false;
    }

    cout << "Depth-first traversal starting from node " << startNode << ": ";
    depthFirstSearch(startNode, numNodes, numEdges);

    return 0;
}
