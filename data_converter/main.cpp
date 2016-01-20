#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

void Tokenize(const std::string& str,
                      std::vector<std::string>& tokens,
                      const std::string& delimiters = " ")
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}


int main(int argc, char **argv) {

	string input_file = "../data/train.txt";
	string output_file = "../data/train_linear.csv";

	int num_sensors = 56;

	int num_lines = -1; // -1 are all lines

	// open input and output file
	ifstream ifs(input_file);
	ofstream ofs(output_file);

	if(ifs.is_open() && ofs.is_open()) {

		// write output file header
		ofs<<"timestamp, SID, count"<<endl;

		string line = "";
		int line_no = 0;
		int buffer[56]; // buffer to hold values for individual sensors
		int timestamp = 0;
		vector<string> tokens;
		while(getline(ifs, line) && (line_no < num_lines || num_lines < 0)) {
			line_no++;
			// Timestamp (DHHMM), S1, S2, S3, S4, S5, S6, S7, S8, S9,
			// S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20,
			// S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31,
			// S32, S33, S34, S35, S36, S37, S38, S39, S40, S41, S42,
			// S43, S44, S45, S46, S47, S48, S49, S50, S51, S52, S53, S54, S55, S56
			if(line_no % 1000 == 0)
				cout<<line_no<<endl;
			// skip header
			if(line_no != 1) {
				
			if(!tokens.empty())tokens.clear();

			Tokenize(line, tokens, ",");
			timestamp = stoi(tokens[0]);
			for(int i = 0; i < num_sensors; i++) {
				buffer[i] = stoi(tokens[i + 1]);
			}

			// output in new format
			// timestamp, SID, count
			for(int i = 0; i < num_sensors; i++) {
				ofs<<timestamp<<","<<(i+1)<<","<<buffer[i]<<endl;
			}
			
			}
		}

		// close files
		ifs.close();
		ofs.close();
	} else {
		cout<<"failed reading or writing file!"<<endl;
	}

	return 0;
}