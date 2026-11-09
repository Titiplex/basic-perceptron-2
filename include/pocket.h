#pragma once

#include "dataset.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace pocket
{
    enum class Initialization { Random, Hebb };

    struct Options
    {
        double eta = 0.6;
        std::size_t max_epochs = 10000;
        std::size_t max_errors = 5;
        std::uint32_t seed = 42;
        bool shuffle = true;
        bool normalize_initial = true;
    };

    struct Model
    {
        std::vector<double> W;

        [[nodiscard]] double score(const std::vector<double>& x) const
        {
            if (W.size() != x.size() + 1)
                throw std::invalid_argument("incorrect model dimensions");
            double s = W[0];
            for (std::size_t j = 0; j < x.size(); ++j) s += W[j + 1] * x[j];
            return s;
        }

        [[nodiscard]] int output(const std::vector<double>& x) const
        {
            return score(x) < 0.0 ? -1 : +1;
        }

        [[nodiscard]] double stability(const std::vector<double>& x, const int target) const
        {
            if (target != -1 && target != 1) throw std::invalid_argument("expected : -1/+1");
            double norm2 = 0.0;
            for (std::size_t j = 1; j < W.size(); ++j) norm2 += W[j] * W[j];
            if (norm2 == 0.0) throw std::runtime_error("null spatial norm");
            return target * score(x) / std::sqrt(norm2);
        }
    };

    inline std::size_t error_count(const Model& model, const Dataset& data)
    {
        std::size_t errors = 0;
        for (const auto& [x, t] : data) errors += model.output(x) != t;
        return errors;
    }

    struct HistoryPoint
    {
        std::size_t epoch;
        std::size_t updates;
        std::size_t current_errors;
        std::size_t best_errors;
    };

    struct Result
    {
        Model best;
        std::size_t best_errors = 0;
        std::size_t epochs_started = 0;
        std::size_t updates = 0;
        bool target_reached = false;
        std::vector<HistoryPoint> history;
    };

    inline void validate(const Dataset& data)
    {
        if (data.empty()) throw std::invalid_argument("Empty dataset");
        const auto n = data.begin()->first.size();
        if (n == 0) throw std::invalid_argument("no characteristic");
        for (const auto& [x, t] : data)
        {
            if (x.size() != n || (t != -1 && t != 1)) throw std::invalid_argument("invalid dimensions or classes");
            for (const double value : x) if (!std::isfinite(value)) throw std::invalid_argument("value not finite");
        }
    }

    inline Model initialize(const Dataset& data, const Initialization init, const std::uint32_t seed,
                            const bool normalize = true)
    {
        validate(data);
        Model model{std::vector(data.begin()->first.size() + 1, 0.0)};
        if (init == Initialization::Random)
        {
            std::mt19937 rng(seed);
            std::uniform_real_distribution dist(-1.0, 1.0);
            for (double& w : model.W) w = dist(rng);
        }
        else
        {
            // Hebb supervised : W = (1/P) sum_mu t_mu * (1, x_mu).
            for (const auto& [x, t] : data)
            {
                model.W[0] += t;
                for (std::size_t j = 0; j < x.size(); ++j)
                    model.W[j + 1] += t * x[j];
            }
            for (double& w : model.W) w /= static_cast<double>(data.size());
        }
        if (normalize)
        {
            double norm2 = 0.0;
            for (const double w : model.W) norm2 += w * w;
            if (norm2 > 0.0) for (double& w : model.W) w /= std::sqrt(norm2);
        }
        return model;
    }

    inline Result fit(const Dataset& train, const Initialization init, const Options& o)
    {
        validate(train);
        if (!std::isfinite(o.eta) || o.eta <= 0.0)
            throw std::invalid_argument("eta must be finite and > 0");
        if (o.max_errors > train.size())
            throw std::invalid_argument("threshold superior to example number");

        Model current = initialize(train, init, o.seed, o.normalize_initial);
        Result r;
        r.best = current;
        r.best_errors = error_count(current, train);
        r.history.push_back({0, 0, r.best_errors, r.best_errors});
        r.target_reached = r.best_errors <= o.max_errors;
        if (r.target_reached) return r;

        std::vector<Dataset::const_iterator> examples;
        for (auto it = train.begin(); it != train.end(); ++it) examples.push_back(it);

        std::mt19937 order_rng(o.seed ^ 0x9e3779b9u);

        for (std::size_t epoch = 1; epoch <= o.max_epochs; ++epoch)
        {
            r.epochs_started = epoch;
            if (o.shuffle) std::ranges::shuffle(examples, order_rng);
            std::size_t errors = error_count(current, train);
            for (const auto it : examples)
            {
                const auto& [x, target] = *it;
                const int prediction = current.output(x);
                if (prediction == target) continue;

                // eta*(t - prediction).
                const double correction = o.eta * (target - prediction);
                current.W[0] += correction;
                for (std::size_t j = 0; j < x.size(); ++j) current.W[j + 1] += correction * x[j];
                for (const double w : current.W) if (!std::isfinite(w)) throw std::runtime_error("weights not finite");
                ++r.updates;

                errors = error_count(current, train);
                if (errors < r.best_errors)
                {
                    r.best = current;
                    r.best_errors = errors;
                }
                if (r.best_errors <= o.max_errors)
                {
                    r.target_reached = true;
                    r.history.push_back({epoch, r.updates, errors, r.best_errors});
                    return r;
                }
            }
            r.history.push_back({epoch, r.updates, errors, r.best_errors});
        }
        return r;
    }
}
