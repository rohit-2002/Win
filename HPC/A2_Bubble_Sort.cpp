#include <iostream>
#include <vector>
#include <omp.h>
#include <ctime>

using namespace std;

void bubbleSort(vector<int> &arr);
void parallelBubbleSort(vector<int> &arr);

int main() {
    int *a,n;
    cout << "Enter the size of the array: ";
    cin >> n;
    a = new int[n];
    for (int i = 0; i < n; i++) {
        cout << "Enter element " << i + 1 << ": ";
        cin >> a[i];
    }
    // Generating random numbers
    srand(time(nullptr));
    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    // Sequential Bubble Sort
    vector<int> bubbleArr = arr;
    double startTime = omp_get_wtime();
    bubbleSort(bubbleArr);
    double endTime = omp_get_wtime();
    cout << "Sequential Bubble Sort Time: " << endTime - startTime << " seconds\n";

    // Parallel Bubble Sort
    vector<int> parallelBubbleArr = arr;
    startTime = omp_get_wtime();
    parallelBubbleSort(parallelBubbleArr);
    endTime = omp_get_wtime();
    cout << "Parallel Bubble Sort Time: " << endTime - startTime << " seconds\n";
    
    cout << "\nSorted array is: ";
    for (int i = 0; i < n; i++) {
        cout << "\n" << a[i];
    }


    return 0;
}

void bubbleSort(vector<int> &arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(vector<int> &arr) {
    int n = arr.size();
    int swapped;
    for (int i = 0; i < n; i++) {
        swapped = 0;
        #pragma omp parallel for shared(arr) reduction(||:swapped)
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = 1;
            }
        }
        if (swapped == 0)
            break;
    }
}
//  g++  A2_Bubble_Sort.cpp -fopenmp
// For Windows ./a.exe
// For Ubuntu ./a.
