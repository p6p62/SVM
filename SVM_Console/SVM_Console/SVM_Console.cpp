#include <iostream>
#include <fstream>
#include "BinaryLinearSVM.h"
#include "CSVReader.h"
#include "ClassifierModelIO.h"

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
	std::ifstream csv_file{ R"(Debug\test_data.txt)" };
	CSVReader::CSV_t<std::string> csv_data_str{ CSVReader::read_csv(csv_file) };
	csv_file.close();
	bool is_right{ CSVReader::is_rows_same_length(csv_data_str) };
	CSVReader::CSV_t<double> csv_data;
	CSVReader::parse_doubles(csv_data_str, csv_data);

	std::vector<TrainingDataVector> data;
	for (const std::vector<double> row : csv_data)
	{
		TrainingDataVector& v{ data.emplace_back() };
		v.data_vector = row;
		v.data_vector.pop_back(); // удаление метки класса
		v.class_label = *row.rbegin() >= 0 ? ClassLabel::FIRST : ClassLabel::SECOND;
	}

	/*std::vector<TrainingDataVector> data;
	get_data(data);*/
	ClassifierModelIO::save_svm_config(BinaryLinearSVM{ data }.get_model(), "model.bin");
	BinaryLinearSVM classifier{ ClassifierModelIO::load_svm_config("model.bin") };

	std::vector<ClassLabel> classification_result;
	for (const TrainingDataVector& x : data)
		classification_result.push_back(classifier.classify(x.data_vector));
	ClassLabel cl{ classifier.classify({10, 15}) };
}
