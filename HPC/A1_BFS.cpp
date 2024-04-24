#include<iostream>
#include<queue>

using namespace std;

class Node {
public:
    Node *left, *right; 
    int data;
};

class BreadthFirstSearch {
public:
    Node *insert(Node *, int); 
    void bfs(Node *);
};

// Function to insert a node into the binary tree
Node *BreadthFirstSearch::insert(Node *root, int data) {
    if (!root) {
        // If the root is null, create a new node and set it as the root
        root = new Node; 
        root->left = NULL; 
        root->right = NULL; 
        root->data = data; 
        return root;
    }

    // Perform level order traversal to find the appropriate position to insert the node
    queue<Node *> q; 
    q.push(root);
    while (!q.empty()) {
        Node *temp = q.front();
        q.pop();
        if (temp->left == NULL) {
            // If the left child is null, insert the new node as the left child
            temp->left = new Node; 
            temp->left->left = NULL; 
            temp->left->right = NULL; 
            temp->left->data = data;
            return root;
        } else {
            // Otherwise, add the left child to the queue for further traversal
            q.push(temp->left);
        }
        if (temp->right == NULL) {
            // If the right child is null, insert the new node as the right child
            temp->right = new Node; 
            temp->right->left = NULL; 
            temp->right->right = NULL; 
            temp->right->data = data; 
            return root;
        } else {
            // Otherwise, add the right child to the queue for further traversal
            q.push(temp->right);
        }
    }
    return root;
}

// Function to perform breadth-first traversal of the binary tree
void BreadthFirstSearch::bfs(Node *root) {
    if (!root) return;

    queue<Node*> q; 
    q.push(root);
    while (!q.empty()) {
        int qSize = q.size(); 
        for (int i = 0; i < qSize; i++) {
            Node* currNode = q.front();
            q.pop();
            cout << "\t" << currNode->data;
            
            if (currNode->left)
                q.push(currNode->left); 
            if (currNode->right)
                q.push(currNode->right);
        }
    }
}

int main() {
    Node *root = NULL; 
    int num_elements;
    cout << "Enter the number of elements you want to insert: ";
    cin >> num_elements;
    
    if (num_elements <= 0) {
        cout << "Invalid number of elements." << endl;
        return 1;
    }
    
    cout << "Enter data for " << num_elements << " elements:\n";
    for (int i = 0; i < num_elements; ++i) {
        int data;
        cout << "Element " << i+1 << ": ";
        cin >> data;
        
        BreadthFirstSearch bfs;
        root = bfs.insert(root, data);
    }

    cout << "Breadth-first traversal: ";
    BreadthFirstSearch bfs;
    bfs.bfs(root);

    return 0;
}
