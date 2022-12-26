#include <iostream>
#include <fstream>
#include "BinaryLinearSVM.h"
#include "CSVReader.h"
#include "ClassifierModelIO.h"

enum class Mode : int
{
	TRAIN,
	CLASSIFY,
	HELP
};

constexpr const char* MODE_ARGS[]{ "--train", "--classify", "--help" };

inline void print_hint()
{
	std::cout
		<< "Приложение-интерфейс для линейного бинарного SVM-классификатора\n"
		<< "\tРежимы\n"
		<< "Обучение:\n\t--train <training_csv_filename> <model_filename> [--no-header]\n"
		<< "Классификация:\n\t--classify <model_filename> <input_csv_filename> <out_filename> [--no-header]\n"
		<< "Справка:\n\t--help или вызов без параметров\n\n"
		<< "Данные предоставляются в csv-формате. Допустима первая строка-заголовок\n"
		<< "По умолчанию первая строка данных отбрасывается. Указание параметра --no-header включает её в обработку\n"
		<< "Сами данные должны представлять собой числа, допустима десятичная точка\n"
		<< "Все строки данных для любого режима должны иметь одинаковое количество элементов\n"
		<< "Данные должны быть линейно-разделимыми. Это приложение не имеет функционала для настройки классификатора\n"
		<< "случае данных, которых нельзя разделить линейно. Однако исходный код классификатора написан с заделом на\n"
		<< "использование функций ядра"
		<< "При обучении модели последним пунктом каждой строки обучающей выборки идёт метка класса, представляющая\n"
		<< "собой 0 или положительное число для объектов первого класса и отрицательное число для объектов второго\n"
		<< "При классификации метка класса в данных должна отсутствовать\n"
		<< "\n\tПример для пространства признаков размерности 3:\n"
		<< "Обучающая выборка\n"
		<< "1,5,8.42,-1\n"
		<< "-4,0.65,-12,1\n"
		<< "2.4,3,7,-1\n"
		<< "Данные для классификации\n"
		<< "4.08,4.5,4\n"
		<< "1,0,0\n\n"
		<< "Выходные данные представляют собой построчный вывод меток классов (1 для первого и -1 для второго) для\n"
		<< "каждого объекта данных. Номер строки данных соответствует номеру строки с ответом классификатора для неё\n"
		<< "Строка с заголовком (в случае её наличия) в выходной файл не копируется\n";
}

bool check_filename(const std::string& filename)
{
	const std::string WRONG_SYMBOLS{ "?\\/:*\"|" };
	return !filename.empty()
		&& std::find_if(filename.begin(), filename.end(),
			[&WRONG_SYMBOLS](char c) -> bool { return WRONG_SYMBOLS.find(c) != -1; }) == filename.end();
}

bool is_file_exists(const std::string& path)
{
	// при несуществующем файле оператор void*() у временного объекта даст nullptr
	return static_cast<bool>(std::ifstream(path));
}

bool check_file_paths(const std::initializer_list<const char*>& files)
{
	bool result{ true };
	for (const char* f : files)
	{
		result = check_filename(f) && is_file_exists(f);
		if (!result)
			break;
	}
	return result;
}



Mode parse_args(int argc, char** argv)
{
	if (argc == 1)
		return Mode::HELP;

	std::string mode_str{ argv[1] };
	Mode mode;
	if (mode_str == MODE_ARGS[static_cast<size_t>(Mode::TRAIN)])
	{
		mode = Mode::TRAIN;
		if (argc != 4 && argc != 5)
			throw std::invalid_argument("Неправильное количество параметров!");
		if (!(check_file_paths({ argv[2] }) && check_filename(argv[3])))
			throw std::invalid_argument("Указанные файлы не существуют или их имена неверны!");
		if (argc == 5 && std::string{ argv[4] } != "--no-header")
			throw std::invalid_argument("Неизвестный последний параметр!");
	}
	else if (mode_str == MODE_ARGS[static_cast<size_t>(Mode::CLASSIFY)])
	{
		mode = Mode::CLASSIFY;
		if (argc != 5 && argc != 6)
			throw std::invalid_argument("Неправильное количество параметров!");
		if (!(check_file_paths({ argv[2], argv[3] }) && check_filename(argv[4])))
			throw std::invalid_argument("Указанные файлы не существуют или их имена неверны!");
		if (argc == 6 && std::string{ argv[5] } != "--no-header")
			throw std::invalid_argument("Неизвестный последний параметр!");
	}
	else if (mode_str == MODE_ARGS[static_cast<size_t>(Mode::HELP)])
	{
		mode = Mode::HELP;
	}
	else
	{
		throw std::invalid_argument("Неправильный режим работы!");
	}
	return mode;
}

bool get_data_vectors_from_csv(const std::string& filename, std::vector<DataVector>& out, bool have_header = true)
{
	std::ifstream csv_file{ filename };
	CSVReader::CSV_t<std::string> csv_data_str{ CSVReader::read_csv(csv_file) };
	csv_file.close();

	bool is_right{ CSVReader::is_rows_same_length(csv_data_str) };
	CSVReader::CSV_t<double> csv_data;
	CSVReader::parse_doubles(csv_data_str, csv_data, have_header);
	out = csv_data;
	return is_right;
}

void train_mode(int argc, char** argv)
{
	bool have_header{ argc != 5 }; // по спецификации после проверки 5 параметров может быть только с --no-header
	std::vector<DataVector> csv_data;
	bool is_right{ get_data_vectors_from_csv(argv[2], csv_data, have_header) };
	if (!is_right)
	{
		std::cerr << "Ошибка! Обучающая выборка имеет неверный формат\n";
		return;
	}

	std::vector<TrainingDataVector> training_data;
	for (const std::vector<double>& row : csv_data)
	{
		TrainingDataVector& v{ training_data.emplace_back() };
		v.data_vector = row;
		v.data_vector.pop_back(); // удаление метки класса
		v.class_label = *row.rbegin() >= 0 ? ClassLabel::FIRST : ClassLabel::SECOND;
	}

	ClassifierModelIO::save_svm_config(BinaryLinearSVM{ training_data }.get_model(), argv[3]);
	std::cout << "Настройка модели завершена успешно!\n";
}

void classify_mode(int argc, char** argv)
{
	// чтение данных
	bool have_header{ argc != 6 }; // по спецификации после проверки 6 параметров может быть только с --no-header
	std::vector<DataVector> csv_data;
	bool is_right{ get_data_vectors_from_csv(argv[3], csv_data, have_header) };
	if (!is_right)
	{
		std::cerr << "Ошибка! Данные имеют неверный формат\n";
		return;
	}

	// загрузка модели и классификация
	BinaryLinearSVM::ClassifierModel model;
	try
	{
		model = ClassifierModelIO::load_svm_config(argv[2]);
	}
	catch (const std::exception&)
	{
		std::cerr << "При загрузке модели возникла ошибка. Возможно, её файл повреждён или имеет неверный формат\n";
		return;
	}
	BinaryLinearSVM classifier{ model };
	std::vector<ClassLabel> classification_result;
	for (const DataVector& x : csv_data)
	{
		classification_result.push_back(classifier.classify(x));
	}

	// запись результата
	std::ofstream f{ argv[4] };
	for (ClassLabel c : classification_result) f << static_cast<int>(c) << std::endl;
	f.close();

	std::cout << "Классификация завершена успешно!\n";
}

void help_mode()
{
	print_hint();
}

int main(int argc, char** argv)
{
	setlocale(LC_ALL, "Russian");
	setlocale(LC_NUMERIC, "English");

	Mode mode;
	try
	{
		mode = parse_args(argc, argv);
		switch (mode)
		{
			case Mode::TRAIN:
				train_mode(argc, argv);
				break;
			case Mode::CLASSIFY:
				classify_mode(argc, argv);
				break;
			case Mode::HELP:
				print_hint();
				break;
			default:
				break;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << "\n";
		return -1;
	}
}
