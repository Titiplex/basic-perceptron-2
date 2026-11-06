#pragma once

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>

using Dataset = std::map<std::vector<int>, int>;

struct SonarDataset
{
    Dataset complete;
    Dataset train;
    Dataset test;
};

constexpr int SONAR_SCALE = 10000;

static void load_sonar_file(
    const std::string& filepath,
    int label,
    size_t expected_count,
    SonarDataset& dataset
)
{
    const auto path = std::filesystem::u8path(filepath);
    std::ifstream file(path);

    if (!file.is_open())
        throw std::runtime_error("Cannot open : " + filepath);

    std::string prefix;
    size_t count = 0;

    while (getline(file, prefix, '{'))
    {
        if (file.eof())
            break;

        const auto end = prefix.find_last_not_of(" \t\r\n");

        if (end == std::string::npos || prefix[end] != ':')
            throw std::runtime_error("invalid id : " + filepath);

        const auto line_start = prefix.find_last_of("\r\n", end);
        const std::string identifier = prefix.substr(
            line_start == std::string::npos ? 0 : line_start + 1);

        const auto first = identifier.find_first_not_of(" \t");
        const bool is_train =
            first != std::string::npos && identifier[first] == '*';

        std::string block;

        if (!getline(file, block, '}') || file.eof())
            throw std::runtime_error("unfinished sonar bloc : " + filepath);

        std::istringstream values(block);
        std::vector<int> features;
        features.reserve(60);

        for (size_t i = 0; i < 60; ++i)
        {
            double value;

            if (!(values >> value) ||
                !std::isfinite(value) || value < 0.0 || value > 1.0)
            {
                throw std::runtime_error(
                    "invalid sonar value " + filepath +
                    ", example " + std::to_string(count + 1));
            }

            features.push_back(
                static_cast<int>(lround(value * SONAR_SCALE)));
        }

        values >> std::ws;

        if (!values.eof())
            throw std::runtime_error(
                "more than 60 values in an example : " + filepath);

        if (!dataset.complete.emplace(features, label).second)
            throw std::runtime_error(
                "duplicated vector after conversion : " + filepath);

        Dataset& subset = is_train ? dataset.train : dataset.test;
        subset.emplace(features, label);

        ++count;
    }

    if (file.bad())
        throw std::runtime_error("reading error : " + filepath);

    if (count != expected_count)
        throw std::runtime_error(
            "incorrect number of examples in " + filepath +
            " : " + std::to_string(count) +
            ", expected : " + std::to_string(expected_count));
}

inline SonarDataset get_dataset()
{
    SonarDataset dataset;

    load_sonar_file(DATASET_DIR "/sonar.mines", +1, 111, dataset);

    load_sonar_file(DATASET_DIR "/sonar.rocks", -1, 97, dataset);

    return dataset;
}
