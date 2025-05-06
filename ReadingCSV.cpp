# include <iostream>
# include <fstream>
# include <vector>
# include <sstream>

using namespace std; 

string FILE_PATH = "datasets/NationalNames.csv";

int main(){
    ifstream file(FILE_PATH);
    if(!file.is_open()){
        cout << "Unable to open file";
        return 1;
    }
    int rowCount = 0;
    vector<string> names;
    string line;
    int NameColumnIndex = -1;

    if(getline(file, line)){
        stringstream ss(line);
        string column;
        int index = 0;
        while(getline(ss, column, ',')){
            if(column == "Name"){
                NameColumnIndex = index;
                break;
            }index++;
        }
        if(NameColumnIndex == -1){
            cout << "'Name' column not found in dataset\n" << endl;
            file.close();
            return 1;
        }
    }
    while(getline(file, line) && rowCount < 10000){
        vector<string> row;
        stringstream ss(line);
        string value;
        while(getline(ss, value, ',')){      // split current line into cols 
            row.push_back(value);
        }
        // example of a value stored in row : 
        // row[i] => 2343(Id) Leonora(Name) 1881(year) F(Gender) 23(Count)
        if(NameColumnIndex < row.size()){    // check if column 'Name' is present in the row 
            names.push_back(row[NameColumnIndex]);
        }
        rowCount += 1;
    }
    cout << "Total names : " << names.size() << endl; 
    file.close();
    return 0;
}