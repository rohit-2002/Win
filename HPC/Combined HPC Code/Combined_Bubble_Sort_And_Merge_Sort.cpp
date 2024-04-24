#include <iostream>
#include <vector>
#include <omp.h>
#include <ctime>

using namespace std;

void bubbleSort(vector<int> &arr);
void parallelBubbleSort(vector<int> &arr);
void mergeSort(int a[], int i, int j);
void merge(int a[], int i1, int j1, int i2, int j2);

int main() {
    int *a, n;
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

    // Sequential Merge Sort
    startTime = omp_get_wtime();
    mergeSort(a, 0, n - 1);
    endTime = omp_get_wtime();
    cout << "\nSequential Merge Sort Time: " << endTime - startTime << " seconds\n";

    // Parallel Merge Sort
    startTime = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergeSort(a, 0, n - 1);
        }
    }
    endTime = omp_get_wtime();
    cout << "Parallel Merge Sort Time: " << endTime - startTime << " seconds\n";

    cout << "\nSorted array is: ";
    for (int i = 0; i < n; i++) {
        cout << "\n" << a[i];
    }

    delete[] a;
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

void mergeSort(int a[], int i, int j) {
    if (i < j) {
        int mid = (i + j) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                mergeSort(a, i, mid);
            }
            #pragma omp section
            {
                mergeSort(a, mid + 1, j);
            }
        }
        merge(a, i, mid, mid + 1, j);
    }
}

void merge(int a[], int i1, int j1, int i2, int j2) {
    int temp[1000];
    int i = i1, j = i2, k = 0;
    while (i <= j1 && j <= j2) {
        if (a[i] < a[j]) {
            temp[k++] = a[i++];
        } else {
            temp[k++] = a[j++];
        }
    }
    while (i <= j1) {
        temp[k++] = a[i++];
    }
    while (j <= j2) {
        temp[k++] = a[j++];
    }
    for (i = i1, j = 0; i <= j2; i++, j++) {
        a[i] = temp[j];
    }
}
// g++ Combined_Bubble_Sort_And_Merge_Sort.cpp -fopenmp
// For Windows ./a.exe
// For Ubuntu ./a.out


// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <chrono>
// #include <omp.h>

// using namespace std;
// using namespace chrono;

// // Function to perform bubble sort
// void bubbleSort(vector<int> &arr) {
//     int n = arr.size();
//     for (int i = 0; i < n - 1; ++i) {
//         for (int j = 0; j < n - i - 1; ++j) {
//             if (arr[j] > arr[j + 1]) {
//                 swap(arr[j], arr[j + 1]);
//             }
//         }
//     }
// }

// // Function to merge two sorted subarrays
// void merge(vector<int> &arr, int l, int m, int r) {
//     int n1 = m - l + 1;
//     int n2 = r - m;
//     vector<int> L(arr.begin() + l, arr.begin() + m + 1);
//     vector<int> R(arr.begin() + m + 1, arr.begin() + r + 1);

//     int i = 0, j = 0, k = l;
//     while (i < n1 && j < n2) {
//         if (L[i] <= R[j]) {
//             arr[k++] = L[i++];
//         } else {
//             arr[k++] = R[j++];
//         }
//     }

//     while (i < n1) {
//         arr[k++] = L[i++];
//     }

//     while (j < n2) {
//         arr[k++] = R[j++];
//     }
// }

// // Function to perform merge sort
// void mergeSort(vector<int> &arr, int l, int r) {
//     if (l < r) {
//         int m = l + (r - l) / 2;
// #pragma omp parallel sections
//         {
// #pragma omp section
//             {
//                 mergeSort(arr, l, m);
//             }
// #pragma omp section
//             {
//                 mergeSort(arr, m + 1, r);
//             }
//         }

//         merge(arr, l, m, r);
//     }
// }

// int main() {
//     int n;
//     cout << "Enter the size of the array: ";
//     cin >> n;

//     vector<int> arr(n);
//     srand(time(0));
//     for (int i = 0; i < n; ++i) {
//         arr[i] = rand() % 100;
//     }

//     vector<int> originalArr = arr;

//     // Sequential Bubble Sort
//     auto start = high_resolution_clock::now();
//     bubbleSort(arr);
//     auto end = high_resolution_clock::now();
//     duration<double> sequentialBubbleTime = end - start;

//     // Parallel Bubble Sort
//     arr = originalArr;
//     start = high_resolution_clock::now();
// #pragma omp parallel
//     {
//         bubbleSort(arr);
//     }
//     end = high_resolution_clock::now();
//     duration<double> parallelBubbleTime = end - start;

//     // Sequential Merge Sort
//     arr = originalArr;
//     start = high_resolution_clock::now();
//     mergeSort(arr, 0, n - 1);
//     end = high_resolution_clock::now();
//     duration<double> sequentialMergeTime = end - start;

//     // Parallel Merge Sort
//     arr = originalArr;
//     start = high_resolution_clock::now();
// #pragma omp parallel
//     {
// #pragma omp single
//         {
//             mergeSort(arr, 0, n - 1);
//         }
//     }
//     end = high_resolution_clock::now();
//     duration<double> parallelMergeTime = end - start;

//     // Performance measurement
//     cout << "Sequential Bubble Sort Time: " << sequentialBubbleTime.count() << " seconds" << endl;
//     cout << "Parallel Bubble Sort Time: " << parallelBubbleTime.count() << " seconds" << endl;
//     cout << "Sequential Merge Sort Time: " << sequentialMergeTime.count() << " seconds" << endl;
//     cout << "Parallel Merge Sort Time: " << parallelMergeTime.count() << " seconds" << endl;

//     return 0;
// }