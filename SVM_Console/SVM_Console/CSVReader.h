#pragma once
#include "Data.h"
#include <istream>
#include <string>
#include <vector>
#include <algorithm>

class CSVReader
{
public:
	template<typename T>
	using CSV_t = matrix_t<T>;

	/// <summary>
	/// Чтение входных данных в формате CSV с разделением по запятой, игнорируя кавычки
	/// Символ перевода строки распознаётся как завершение текущей строки и начало новой
	/// </summary>
	/// <param name="input">Входной поток данных</param>
	/// <returns>Вектор векторов данных, разделённых по запятым</returns>
	static CSV_t<std::string> read_csv(std::istream& input);

	template <typename T>
	static bool is_rows_same_length(const CSV_t<T>& csv_data) noexcept;
};

template<typename T>
inline bool CSVReader::is_rows_same_length(const CSV_t<T>& csv_data) noexcept
{
	bool result{ csv_data.empty() };
	if (!result)
	{
		const size_t csv_row_length{ csv_data[0].size() };
		result = std::all_of(csv_data.begin(), csv_data.end(), [csv_row_length](const auto& row) -> bool {return row.size() == csv_row_length; });
	}
	return result;
}
