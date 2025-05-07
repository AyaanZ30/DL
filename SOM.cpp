# include <iostream>
# include <vector>
# include <cmath>
# include <algorithm> 
# include <cstdlib>
#include <ctime>   // For time()
#include <limits>  
#include <iomanip>

using namespace std;

double euclidean_distance(const vector<double> &input, const vector<double> &weights){
    double d = 0.0;
    for(const auto x : input){
        for(const auto y : weights){
            d += pow((x - y), 2);
        }
    }return d;
}
void initialize_weights(vector<vector<double>> &KSOM, const int input_units, const int output_units){
    srand(static_cast<unsigned int>(time(0)));
    for(int i = 0 ; i < input_units ; i++){
        for(int j = 0 ; j < output_units ; j++){
            KSOM[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
}
vector<double> get_neuron_weights(vector<vector<double>> KSOM, int output_neuron, int input_units){
    vector<double> weights;
    weights.reserve(input_units);
    for(int i = 0 ; i < input_units ; i++){
        weights.push_back(KSOM[i][output_neuron]);
    }return weights;    
}
void print_weights(vector<vector<double>> &KSOM, const int input_units, const int output_units){
    cout << fixed << setprecision(2);
    cout << "Current KSOM Weights (KSOM[input_feature_idx][output_neuron_idx]) : " << endl;
    for(int j = 0 ; j < output_units ; j++){
        cout << "Neuron " << j+1 << " [ ";
        for(int i = 0 ; i < input_units ; i++){
            cout << KSOM[i][j] << " ";
        }cout << " ]" << endl;
    }cout << defaultfloat;
}
int find_bmu(vector<double> &current_input_vector, vector<vector<double>> KSOM, int input_units, int output_units){
    int bmu_idx = -1;
    double min = 100000.0;
    for(int j = 0 ; j < output_units ; j++){
        vector<double> current_neuron_weights = get_neuron_weights(KSOM, j, input_units);
        double dist = euclidean_distance(current_input_vector, current_neuron_weights);
        if(dist < min){
            min = dist;
            bmu_idx = j;
        }
    }return bmu_idx;
}
void update_weights(vector<double> &curr_input_vector, vector<vector<double>> KSOM, int bmu_idx, double learning_rate, int input_units){
    if(bmu_idx < 0){
        return;
    }
    for(int i = 0 ; i < input_units ; i++){
        KSOM[i][bmu_idx] += (learning_rate)*(curr_input_vector[i] - KSOM[i][bmu_idx]);
    }
}
int main(){
    int input_units, output_units;
    cout << "Enter number of input units (xi) : ";
    cin >> input_units;
    cout << "Enter number of output units (yi) : ";
    cin >> output_units;

    vector<vector<double>> KSOM(input_units, vector<double>(output_units, 0.0));
    initialize_weights(KSOM, input_units, output_units);
    print_weights(KSOM, input_units, output_units);

    vector<vector<double>> training_data = {
        {1.0, 0.0}, {0.0, 1.0}
    };
    int epochs = 50;
    double lr = 0.2;

    for(int epoch = 0 ; epoch < epochs ; epochs++){
        for(auto &sample : training_data){
            int bmu_idx = find_bmu(sample, KSOM, input_units, output_units);
            if(bmu_idx != -1){
                update_weights(sample, KSOM, bmu_idx, lr, input_units);
            }
        }
    }

    cout << "Final weights after training : \n";
    print_weights(KSOM, input_units, output_units);
    return 0;
}