#include <vector>
#include <map>
#include <random>
#include <ranges>

#include "perceptron.h"

using namespace std;

int heaviside(const double x)
{
    return x < 0 ? -1 : 1;
}

double sum(const vector<double>& weights, const vector<int>& input)
{
    double sum = 0;
    for (int i = 0; i < weights.size(); i++)
        sum += weights[i] * input[i];

    return sum;
}

vector<double> add(const vector<double>& weights, const vector<double>& delta)
{
    vector<double> result;
    for (int i = 0; i < weights.size(); i++)
        result.push_back(weights[i] + delta[i]);
    return result;
}

bool equals(const vector<int>& a, const vector<int>& b)
{
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}


int Perceptron::output(const vector<int>& input) const
{
    if (input.size() != this->weights.size())
    {
        throw invalid_argument("Input size does not match number of input weights.");
    }

    return heaviside(sum(this->weights, input) + this->biais);
}

bool Perceptron::verify() const
{
    // building output vector for learn data (correct version)
    vector<int> output_vector;
    for (const auto& val : this->learn_data | views::values)
    {
        output_vector.push_back(val);
    }

    // building current output vector (might be false)
    vector<int> current_output;
    for (const auto& key : this->learn_data | views::keys)
    {
        current_output.push_back(this->output(key));
    }

    if (equals(current_output, output_vector)) return true;
    return false;
}

size_t Perceptron::learn_incremental(const size_t max_epochs)
{
    for (size_t epoch = 0; epoch < max_epochs; epoch++)
    {
        if (verify()) return epoch;
        for (const auto& [fst, snd] : this->learn_data)
        {
            auto x = fst;
            const int y = this->output(fst);

            if (const int t = snd; y != t)
            {
                const double correction = this->eta * static_cast<float>(t - y);

                for (size_t i = 0; i < this->weights.size(); ++i)
                    this->weights[i] += correction * x[i];
                this->biais += correction;
            }
        }
    }
    return verify() ? max_epochs : -1;
}

size_t Perceptron::learn_batch(const size_t max_epochs)
{
    for (size_t epoch = 0; epoch < max_epochs; epoch++)
    {
        if (verify()) return epoch;
        vector delta(this->weights.size(), 0.0);
        double delta_biais = 0;

        for (const auto& [fst, snd] : this->learn_data)
        {
            auto x = fst;
            const int y = this->output(fst);

            if (const int t = snd; y != t)
            {
                vector<double> delta_local;
                for (const auto x_prime : x)
                {
                    delta_local.push_back(this->eta * static_cast<float>(t - y) * static_cast<float>(x_prime));
                }
                delta = add(delta, delta_local);
                delta_biais += this->eta * static_cast<float>(t - y);
            }
        }

        this->weights = add(this->weights, delta);
        this->biais += delta_biais;
    }
    return verify() ? max_epochs : -1;
}

size_t Perceptron::learn(const LearningType type_, const size_t max_epochs)
{
    switch (type_)
    {
    case BATCH:
        return learn_batch(max_epochs);
    default:
        return learn_incremental(max_epochs);
    }
}
