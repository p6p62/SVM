#include <iostream>
#include <fstream>
#include "BinaryLinearSVM.h"
#include "CSVReader.h"

using ClassLabel::FIRST;
using ClassLabel::SECOND;

constexpr void get_data(std::vector<TrainingDataVector>& out_data)
{
	out_data.push_back({ {2, 0}, FIRST });
	out_data.push_back({ {1, 1}, FIRST });
	out_data.push_back({ {0, 5}, FIRST });
	out_data.push_back({ {1.66, 2.5}, FIRST });
	out_data.push_back({ {1.92, 1.68}, FIRST });
	out_data.push_back({ {-0.8, 1.64}, SECOND });
	out_data.push_back({ {-0.5, 0}, SECOND });
}

int main()
{
	std::ifstream csv_file{ "csv_data.txt" };
	CSVReader::CSV_t<std::string> csv_data{ CSVReader::read_csv(csv_file) };
	csv_file.close();
	bool is_right{ CSVReader::is_rows_same_length(csv_data) };


	std::vector<TrainingDataVector> data;
	get_data(data);
	BinaryLinearSVM classifier{ data };

	std::vector<ClassLabel> classification_result;
	for (const TrainingDataVector& x : data)
		classification_result.push_back(classifier.classify(x.data_vector));
	ClassLabel cl{ classifier.classify({0, 1.5}) };
}
