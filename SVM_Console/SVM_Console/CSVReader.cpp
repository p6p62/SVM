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

void CSVReader::parse_doubles(const CSV_t<std::string>& in, CSV_t<double>& out, bool have_header)
{
	size_t index_offset{ have_header && 1 };
	if (out.size() < in.size())
		out.resize(in.size() - index_offset);
	for (size_t i{ index_offset }, o{ 0 }; i < in.size(); ++i, ++o)
	{
		if (out[o].size() < in[i].size())
			out[o].resize(in[i].size());
		for (size_t j{ 0 }; j < in[i].size(); ++j)
			out[o][j] = std::stod(in[i][j]);
	}
}
