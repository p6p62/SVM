#include "ClassifierModelIO.h"
#include <stdexcept>
#include <fstream>

template<typename T>
static constexpr void write_bytes(const T* data, size_t count, std::vector<char>& data_buffer) noexcept
{
	const auto DATA_BYTE_SIZE{ sizeof(T) * count };
	const size_t OLD_SIZE{ data_buffer.size() };
	data_buffer.resize(OLD_SIZE + DATA_BYTE_SIZE);
	memcpy(data_buffer.data() + OLD_SIZE, data, DATA_BYTE_SIZE);
}

template<typename T>
static constexpr void write_bytes(const T& value, std::vector<char>& data_buffer) noexcept
{
	write_bytes(&value, 1, data_buffer);
}

template<typename T>
static constexpr void read_from_bytes(T* out_data, const std::vector<char>& data_buffer, size_t& start_position, size_t count)
{
	const auto DATA_BYTE_SIZE{ sizeof(T) * count };
	if (DATA_BYTE_SIZE + start_position > data_buffer.size())
		throw std::length_error("Ошибка! Нет такого количества данных");
	memcpy(out_data, data_buffer.data() + start_position, DATA_BYTE_SIZE);
	start_position += DATA_BYTE_SIZE;
}

template<typename T>
static constexpr void read_from_bytes(T& value, const std::vector<char>& data_buffer, size_t& start_position)
{
	read_from_bytes(&value, data_buffer, start_position, 1);
}

void ClassifierModelIO::save_svm_config(const BinaryLinearSVM::ClassifierModel& model, const std::string& filename)
{
	// преобразование в байты
	std::vector<char> model_bytes;
	write_bytes<uint64_t>(model.dimensionality, model_bytes);
	write_bytes<char>(static_cast<char>(model.kernel_type), model_bytes);
	write_bytes<double>(model.offset, model_bytes);
	write_bytes<uint64_t>(model.support_vectors.size(), model_bytes);
	for (const std::pair<TrainingDataVector, number_elem_t>& p : model.support_vectors)
	{
		write_bytes<double>(p.first.data_vector.data(), model.dimensionality, model_bytes);
		write_bytes<char>(static_cast<char>(p.first.class_label), model_bytes);
		write_bytes<double>(p.second, model_bytes);
	}

	// запись в файл
	std::ofstream bin_file{ filename, std::ios::binary };
	bin_file.write(model_bytes.data(), model_bytes.size());
	bin_file.close();
}

bool read_full_file(const std::string& filename, std::vector<char>& out_data)
{
	std::ifstream bin_file{ filename, std::ios::in | std::ios::binary | std::ios::ate };
	bool result{ bin_file.is_open() };
	if (result)
	{
		std::streampos file_size{ bin_file.tellg() };
		result = file_size >= 0;
		if (result)
		{
			out_data.resize(file_size);
			bin_file.seekg(0, std::ios::beg);
			bin_file.read(out_data.data(), file_size);
			result = bin_file.good();
			bin_file.close();
		}
	}
	return result;
}

inline constexpr void select_kernel(kernel_t& kernel, BinaryLinearSVM::KernelType kernel_type)
{
	switch (kernel_type)
	{
		case BinaryLinearSVM::KernelType::SCALAR_PRODUCT:
			kernel = scalar_product;
			break;
		default:
			kernel = scalar_product;
			break;
	}
}

BinaryLinearSVM::ClassifierModel ClassifierModelIO::load_svm_config(const std::string& filename)
{
	// чтение из файла
	std::vector<char> model_bytes;
	read_full_file(filename, model_bytes);

	// преобразование из байт
	BinaryLinearSVM::ClassifierModel model;
	size_t memory_position{ 0 };

	// размерность
	uint64_t dimensionality;
	read_from_bytes<uint64_t>(dimensionality, model_bytes, memory_position);
	model.dimensionality = static_cast<size_t>(dimensionality);

	// тип ядра
	char kernel_type;
	read_from_bytes<char>(kernel_type, model_bytes, memory_position);
	model.kernel_type = static_cast<BinaryLinearSVM::KernelType>(kernel_type);
	select_kernel(model.kernel, model.kernel_type);

	// смещение
	read_from_bytes<double>(model.offset, model_bytes, memory_position);

	// опорные векторы
	uint64_t support_vectors_count;
	read_from_bytes<uint64_t>(support_vectors_count, model_bytes, memory_position);

	for (size_t i{ 0 }; i < static_cast<size_t>(support_vectors_count); ++i)
	{
		std::pair<TrainingDataVector, number_elem_t>& p{ model.support_vectors.emplace_back() };
		DataVector& data_v{ p.first.data_vector };
		data_v.resize(model.dimensionality);
		read_from_bytes<double>(data_v.data(), model_bytes, memory_position, model.dimensionality);
		read_from_bytes<char>(reinterpret_cast<char*>(&p.first.class_label), model_bytes, memory_position, 1);
		read_from_bytes<double>(p.second, model_bytes, memory_position);
	}

	return model;
}