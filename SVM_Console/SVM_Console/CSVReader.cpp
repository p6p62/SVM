#include "CSVReader.h"
#include <sstream>

CSVReader::CSV_t<std::string> CSVReader::read_csv(std::istream& input)
{
	std::vector<std::vector<std::string>> out;
	std::string csv_row;
	std::istringstream row_stream;
	while (std::getline(input, csv_row))
	{
		row_stream.clear();
		row_stream.str(csv_row);
		std::vector<std::string>& current_csv_row{ out.emplace_back() };
		while (std::getline(row_stream, csv_row, ','))
		{
			current_csv_row.push_back(csv_row);
		}
	}
	return out;
}
