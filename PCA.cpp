# include <iostream> 
# include <cmath> 
# include <algorithm>
# include <vector>
 
using namespace std;
using Matrix = vector<vector<double>>;


vector<double> compute_means(const Matrix &M){
    vector<double> means(M[0].size(), 0.0);
    for(int i = 0 ; i < M[0].size() ; i++){
        for(int j = 0 ; j < M.size() ; j++){
            means[i] += M[j][i];
        }means[i] /= M.size();
    }return means;
}
vector<double> compute_std(const Matrix &M, vector<double> means){
    vector<double> std_dev(M[0].size(), 0.0);
    for(int i = 0 ; i < M[0].size() ; i++){
        for(int j = 0 ; j < M.size() ; j++){
            double cur_mean = means[i];
            std_dev[i] += (M[j][i] - cur_mean)*(M[j][i] - cur_mean);
        }
        std_dev[i] = pow((std_dev[i] / (M.size() - 1)), 0.5);
    }return std_dev; 
}
Matrix standardize_data(Matrix M, vector<double> means, vector<double> std_dev){
    Matrix Y = M;
    int rows = M.size(), cols = M[0].size();
    for(int i = 0 ; i < cols ; i++){
        for(int j = 0 ; j < rows ; j++){
            Y[j][i] = (M[j][i] - means[i]) / (std_dev[i]); 
        }
    }return Y;
}
Matrix covariance_matrix(const Matrix &X){
    int n_samples = X.size();
    int n_features = X[0].size();
    Matrix cov(2, vector<double>(2, 0.0));
    for(int i = 0 ; i < n_features ; i++){
        cov[0][0] += X[i][0] * X[i][0];
        cov[1][0] += X[i][1] * X[i][0];
        cov[0][1] += X[i][0] * X[i][1];
        cov[1][1] += X[i][1] * X[i][1]; 
    }
    for(int i = 0 ; i < 2 ; i++){
        for(int j = 0 ; j < 2 ; j++){
            cov[i][j] = cov[i][j] / (n_samples - 1);
        }
    }return cov;
} 
void EigenDecomposition(Matrix A, vector<double> eigen_values, Matrix eigen_vectors){
    double a = A[0][0], b = A[0][1], c = A[1][1];
    double det = (a*c)-(b*b);
    double trace = (a+c);
    double term = sqrt(trace * trace / 4.0 - det);

    double lambda_1 = trace / 2.0 + term;
    double lambda_2 = trace / 2.0 - term;
    eigen_values = {lambda_1, lambda_2};

    double v1_x = 1.0;
    double v1_y = (lambda_1 - a) / b;
    double norm1 = sqrt(pow(v1_x, 2)+pow(v1_y, 2));

    double v2_x = 1.0;
    double v2_y = (lambda_2 - a) / b;
    double norm2 = sqrt(pow(v2_x, 2)+pow(v2_y, 2));

    eigen_vectors = {
        {v1_x / norm1, v1_y / norm1},{v2_x / norm2, v2_y / norm2}
    };
}
Matrix project_data(const Matrix &X, const Matrix &eigen_vectors, int k){
    Matrix x_proj(X.size(), vector<double>(k, 0.0));
    for(int i = 0 ; i < X.size() ; i++){
        for(int j = 0 ; j < k ; j++){
            for(int d = 0 ; d < X[0].size() ; d++){
                x_proj[i][j] += X[i][d] * eigen_vectors[d][j];
            }
        }
    }return x_proj;
}
void print_matrix(const Matrix& M, const string& label) {
    cout << "\n" << label << ":\n";
    for (const auto& row : M) {
        for (double val : row) {
            cout << val << "\t";
        }
        cout << endl;
    }
}

int main(){
    Matrix X = {
        {2.5, 2.4}, 
        {0.5, 0.7}, 
        {2.2, 2.9}, 
        {1.9, 2.2},
        {3.1, 3.0}, 
        {2.3, 2.7}, 
        {2.0, 1.6}, 
        {1.0, 1.1},
        {1.5, 1.6}, 
        {1.1, 0.9}
    };

    vector<double> means = compute_means(X);

    vector<double> std_dev = compute_std(X, means);

    Matrix X_standardized = standardize_data(X, means, std_dev);

    Matrix cov = covariance_matrix(X_standardized);

    vector<double> eigen_values;
    Matrix eigen_vectors;
    EigenDecomposition(cov ,eigen_values, eigen_vectors);

    Matrix X_proj = project_data(X_standardized, eigen_vectors, 1);
    print_matrix(X_proj, "Projected Data (1D)");

    return 0;
}