#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

using Dataset = map<vector<int>, int>;

struct SonarDataset
{
    Dataset complete;
    Dataset train;
    Dataset test;
};

constexpr int SONAR_SCALE = 10000;

static void load_sonar_file(
    const string& filepath,
    int label,
    size_t expected_count,
    SonarDataset& dataset)
{
    ifstream file(filepath);

    if (!file.is_open())
        throw runtime_error("Cannot open : " + filepath);

    string prefix;
    size_t count = 0;

    while (getline(file, prefix, '{'))
    {
        if (file.eof())
            break;

        const auto end = prefix.find_last_not_of(" \t\r\n");

        if (end == string::npos || prefix[end] != ':')
            throw runtime_error("invalid id : " + filepath);

        const auto line_start = prefix.find_last_of("\r\n", end);
        const string identifier = prefix.substr(
            line_start == string::npos ? 0 : line_start + 1);

        const auto first = identifier.find_first_not_of(" \t");
        const bool is_train =
            first != string::npos && identifier[first] == '*';

        string block;

        if (!getline(file, block, '}') || file.eof())
            throw runtime_error("unfinished sonar bloc : " + filepath);

        istringstream values(block);
        vector<int> features;
        features.reserve(60);

        for (size_t i = 0; i < 60; ++i)
        {
            double value;

            if (!(values >> value) ||
                !isfinite(value) || value < 0.0 || value > 1.0)
            {
                throw runtime_error(
                    "invalid sonar value " + filepath +
                    ", example " + to_string(count + 1));
            }

            features.push_back(
                static_cast<int>(lround(value * SONAR_SCALE)));
        }

        values >> ws;

        if (!values.eof())
            throw runtime_error(
                "more than 60 values in an example : " + filepath);

        if (!dataset.complete.emplace(features, label).second)
            throw runtime_error(
                "duplicated vector after conversion : " + filepath);

        Dataset& subset = is_train ? dataset.train : dataset.test;
        subset.emplace(features, label);

        ++count;
    }

    if (file.bad())
        throw runtime_error("reading error : " + filepath);

    if (count != expected_count)
        throw runtime_error(
            "incorrect number of examples in " + filepath +
            " : " + to_string(count) +
            ", expected : " + to_string(expected_count));
}

SonarDataset get_dataset()
{
    SonarDataset dataset;

    load_sonar_file(DATASET_DIR "/sonar.mines", +1, 111, dataset);

    load_sonar_file(DATASET_DIR "/sonar.rocks", -1, 97, dataset);

    return dataset;
}
