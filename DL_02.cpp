# include <iostream>
# include <vector> 
# include <cmath> 

using namespace std;

double calculate_net_value(const double w1, const double x1, const double w2, const double x2, const double bias){
    double net = ((w1 * x1) + (w2 * x2) + bias);
    return net;
}
double sigmoid_activation(const double weighted_sum){
    double power = (-1 * weighted_sum);
    double layer_output = (1.0 /( 1.0 + exp(power)));
    return layer_output;  
}
double sigmoid_derivative(const double output){
    return ((output) * (1 - output));
}
double total_error_derivative(const double actual_output, const double target_output){
    return (actual_output - target_output);
}
double evaluate_individual_error(const double target_output, const double actual_output){
    double error = ((0.5) * pow((target_output - actual_output), 2));
    return error;
}
double evaluate_total_error(const double error_1, const double error_2){
    return (error_1 + error_2);
}
double differentiate_x_power_n(const double x, const double n){
    return (double)(n * pow(x, (n-1)));
}

void calculate_hidden_layer_output(double i1, double i2, double w1, double w2, double w3, double w4, double b1, double &output_h1, double &output_h2){
    double net_h1 = calculate_net_value(w1, i1, w3, i2, b1);
    output_h1 = sigmoid_activation(net_h1);

    double net_h2 = calculate_net_value(w2, i1, w4, i2, b1);
    output_h2 = sigmoid_activation(net_h2);
}

void calculate_final_layer_output(double h1, double h2, double w5, double w6, double w7, double w8, double b2, double &output_o1, double &output_o2){
    double net_o1 = calculate_net_value(w5, h1, w6, h2, b2);
    output_o1 = sigmoid_activation(net_o1);

    double net_o2 = calculate_net_value(w7, h1, w8, h2, b2);
    output_o2 = sigmoid_activation(net_o2);
}

// D(E_total) / D(o1)   &&  D(E_total) / D(o2) 
void calculate_output_layer_gradients(double target_o1, double target_o2, double output_o1, double output_o2, double output_h1, double output_h2, double &dE_total_dw5, double &dE_total_dw6, double &dE_total_dw7, double &dE_total_dw8){
    double dE_total_doutput_o1 = total_error_derivative(output_o1, target_o1);
    double doutput_o1_dsum_o1 = sigmoid_derivative(output_o1);
    double dsum_o1_dw5 = output_h1;
    double dsum_o1_dw6 = output_h2;

    dE_total_dw5 = (dE_total_doutput_o1 * doutput_o1_dsum_o1 * dsum_o1_dw5);
    dE_total_dw6 = (dE_total_doutput_o1 * doutput_o1_dsum_o1 * dsum_o1_dw6);

    double dE_total_doutput_o2 = total_error_derivative(output_o2, target_o2);
    double doutput_o2_dsum_o2 = sigmoid_derivative(output_o2);
    double dsum_o2_dw7 = output_h1;
    double dsum_o2_dw8 = output_h2;

    dE_total_dw7 = (dE_total_doutput_o2 * doutput_o2_dsum_o2 * dsum_o2_dw7);
    dE_total_dw8 = (dE_total_doutput_o2 * doutput_o2_dsum_o2 * dsum_o2_dw8);
}

// D(E_total) / D(h1)   &&  D(E_total) / D(h2)
void calculate_hidden_layer_gradients(double i1, double i2, double output_h1, double output_h2, double w5, double w6, double w7, double w8, double dE_total_doutput_o1, double doutput_o1_dsum_o1, double dE_total_doutput_o2, double doutput_o2_dsum_o2, double &dE_total_dw1, double &dE_total_dw2, double &dE_total_dw3, double &dE_total_dw4){
    double dE_total_doutput_h1 = (dE_total_doutput_o1 *  doutput_o1_dsum_o1 * w5) + (dE_total_doutput_o2 *  doutput_o2_dsum_o2 * w7);
    double doutput_h1_dsum_h1 = sigmoid_derivative(output_h1);
    double dsum_h1_dw1 = i1;
    double dsum_h1_dw2 = i2;

    dE_total_dw1 = (dE_total_doutput_h1 * doutput_h1_dsum_h1 * dsum_h1_dw1);
    dE_total_dw2 = (dE_total_doutput_h1 * doutput_h1_dsum_h1 * dsum_h1_dw2);

    double dE_total_doutput_h2 = (dE_total_doutput_o1 *  doutput_o1_dsum_o1 * w6) + (dE_total_doutput_o2 *  doutput_o2_dsum_o2 * w8);
    double doutput_h2_dsum_h2 = sigmoid_derivative(output_h2);
    double dsum_h2_dw3 = i1;
    double dsum_h2_dw4 = i2;

    dE_total_dw3 = (dE_total_doutput_h2 * doutput_h2_dsum_h2 * dsum_h2_dw3);
    dE_total_dw4 = (dE_total_doutput_h2 * doutput_h2_dsum_h2 * dsum_h2_dw4);
}

void update_weights(const double learning_rate, double &w1, double &w2, double &w3 ,double &w4, double &w5, double &w6, double &w7 ,double &w8 ,double dE_total_dw1,double dE_total_dw2,double dE_total_dw3,double dE_total_dw4,double dE_total_dw5,double dE_total_dw6,double dE_total_dw7,double dE_total_dw8){
    w1 -= (learning_rate * dE_total_dw1);
    w2 -= (learning_rate * dE_total_dw2);
    w3 -= (learning_rate * dE_total_dw3);
    w4 -= (learning_rate * dE_total_dw4);
    w5 -= (learning_rate * dE_total_dw5);
    w6 -= (learning_rate * dE_total_dw6);
    w7 -= (learning_rate * dE_total_dw7);
    w8 -= (learning_rate * dE_total_dw8);
}
int main(){
    // initialize_variables()
    double w1 = 0.1, w2 = 0.2, w3 = 0.3, w4 = 0.4;
    double w5 = 0.5, w6 = 0.6, w7 = 0.7, w8 = 0.8;
    double b1 = 0.25, b2 = 0.35;
    
    double i1 = 0.1, i2 = 0.5;
    double target_o1 = 0.05, target_o2 = 0.95;
    double learning_rate = 0.6;

    double output_h1, output_h2, output_o1, output_o2;
    calculate_hidden_layer_output(i1, i2, w1, w2, w3, w4, b1, output_h1, output_h2);
    calculate_final_layer_output(output_h1, output_h2, w5, w6, w7, w8, b2, output_o1, output_o2);

    double E1 = evaluate_individual_error(target_o1, output_o1);
    double E2 = evaluate_individual_error(target_o2, output_o2);
    double E_total = evaluate_total_error(E1, E2);

    double dE_total_dw5, dE_total_dw6, dE_total_dw7, dE_total_dw8;
    calculate_output_layer_gradients(target_o1, target_o2, output_o1, output_o2, output_h1, output_h2, dE_total_dw5, dE_total_dw6, dE_total_dw7, dE_total_dw8);

    double dE_total_doutput_o1 = (target_o1 - output_o1);
    double doutput_o1_dsum_o1 = sigmoid_derivative(output_o1);
    double dE_total_doutput_o2 = (target_o2 - output_o2);
    double doutput_o2_dsum_o2 = sigmoid_derivative(output_o2);

    double dE_total_dw1, dE_total_dw2, dE_total_dw3, dE_total_dw4;
    calculate_hidden_layer_gradients(i1, i2, output_h1, output_h2, w5, w6, w7, w8, dE_total_doutput_o1, doutput_o1_dsum_o1, dE_total_doutput_o2, doutput_o2_dsum_o2, dE_total_dw1, dE_total_dw2, dE_total_dw3, dE_total_dw4);

    update_weights(learning_rate, w1, w2, w3, w4, w5, w6, w7, w8, dE_total_dw1, dE_total_dw2, dE_total_dw3, dE_total_dw4, dE_total_dw5, dE_total_dw6, dE_total_dw7, dE_total_dw8);

    cout << "New weights after update:" << endl;
    cout << "w1: " << w1 << endl;
    cout << "w2: " << w2 << endl;
    cout << "w3: " << w3 << endl;
    cout << "w4: " << w4 << endl;
    cout << "w5: " << w5 << endl;
    cout << "w6: " << w6 << endl;
    cout << "w7: " << w7 << endl;
    cout << "w8: " << w8 << endl;

    return 0;
}