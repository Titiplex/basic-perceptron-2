#pragma once

#include <vector>
#include <map>
#include <random>

class Perceptron
{
    std::vector<double> weights;
    float eta;
    double biais;
    std::map<std::vector<double>, int> learn_data;

public:
    enum LearningType
    {
        ONLINE,
        BATCH
    };

    [[nodiscard]] const std::vector<double>& get_weights() const
    {
        return this->weights;
    }

    [[nodiscard]] double get_bias() const
    {
        return biais;
    }

    Perceptron()
    {
        this->weights = {};
        this->eta = 1.0;
        this->biais = 1.0;
        this->learn_data = {};
    }

    Perceptron(const std::vector<double>& weights, const float eta, const double biais,
               const std::map<std::vector<double>, int>& learn_data) :
        learn_data(learn_data)
    {
        this->weights = weights;
        this->eta = eta;
        this->biais = biais;
    }

    Perceptron(const std::vector<double>& weights, const float eta, double biais)
    {
        this->weights = weights;
        this->eta = eta;
        this->biais = biais;
        this->learn_data = {};
    }

    Perceptron(const int n, const float eta, const double biais,
               const std::map<std::vector<double>, int>& learn_data) :
        learn_data(learn_data)
    {
        std::vector<double> init(n);

        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution weight_dist(-1.0, 1.0);

        for (double& w : init) { w = weight_dist(gen); }

        this->weights = init;
        this->eta = eta;
        this->biais = biais;
    }

    Perceptron(const int n, const float eta, const double biais)
    {
        std::vector<double> init(n);

        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution weight_dist(-1.0, 1.0);

        for (double& w : init) { w = weight_dist(gen); }

        this->weights = init;
        this->eta = eta;
        this->biais = biais;
        this->learn_data = {};
    }

    [[nodiscard]] int output(const std::vector<double>& input) const;
    [[nodiscard]] bool verify() const;
    std::size_t learn(LearningType type_, std::size_t max_epochs = 1000);
    [[nodiscard]] double stability(const std::vector<double>& input, int target) const;

private:
    std::size_t learn_incremental(std::size_t max_epochs);
    std::size_t learn_batch(std::size_t max_epochs);
};
