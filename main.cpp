#include <iostream>
#include <matplot/matplot.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>
#include <utility>

#include "dataset.h"
#include "perceptron.h"
#include "pocket.h"
using namespace std;

double error_rate(const Perceptron& model, const Dataset& data)
{
    if (data.empty()) throw invalid_argument("Dataset vide");

    size_t errors = 0;

    for (const auto& [x, target] : data)
        if (model.output(x) != target)
            ++errors;

    return static_cast<double>(errors) / static_cast<double>(data.size());
}

void draw_stability(
    const Dataset& test_data,
    const Dataset& learn_data,
    const Perceptron& perceptron,
    const string& filename
)
{
    vector<double> indices;
    vector<double> gammas;

    indices.reserve(test_data.size());
    gammas.reserve(test_data.size());

    size_t index = 0;
    size_t errors = 0;

    cout << endl << "Exemple\tClasse\tPrediction\tStabilite" << endl;

    for (const auto& [x, target] : test_data)
    {
        ++index;

        const int prediction = perceptron.output(x);
        const double gamma = perceptron.stability(x, target);

        indices.push_back(static_cast<double>(index));
        gammas.push_back(gamma);

        if (prediction != target)
            ++errors;

        cout << index << '\t'
            << target << '\t'
            << prediction << '\t'
            << gamma << endl;
    }

    cout << "Erreurs sur test : " << errors << '/' << test_data.size() << endl;

    if (!learn_data.empty() && !test_data.empty())
    {
        struct PlotGroup
        {
            vector<double> indices;
            vector<double> distances;
        };

        PlotGroup train_mines;
        PlotGroup train_rocks;
        PlotGroup test_mines;
        PlotGroup test_rocks;

        size_t position = 0;

        const auto collect = [&](
            const Dataset& data,
            const int label,
            PlotGroup& group
        )
        {
            for (const auto& [x, target] : data)
            {
                if (target != label) continue;

                ++position;

                // gamma = target * distance
                // donc distance = target * gamma, car target vaut -1 ou +1.
                const double distance = target * perceptron.stability(x, target);

                group.indices.push_back(static_cast<double>(position));
                group.distances.push_back(distance);
            }
        };

        // Ordre : train puis test ; mines puis roches dans chaque ensemble.
        collect(learn_data, +1, train_mines);
        collect(learn_data, -1, train_rocks);

        const double separation = static_cast<double>(position) + 0.5;

        collect(test_data, +1, test_mines);
        collect(test_data, -1, test_rocks);

        // Bornes verticales pour tracer la séparation train/test.
        double y_min = 0.0;
        double y_max = 0.0;

        for (const PlotGroup* group :
             {&train_mines, &train_rocks, &test_mines, &test_rocks})
        {
            for (const double distance : group->distances)
            {
                if (distance < y_min) y_min = distance;

                if (distance > y_max) y_max = distance;
            }
        }

        const double padding = y_max > y_min ? 0.1 * (y_max - y_min) : 0.1;

        y_min -= padding;
        y_max += padding;

        matplot::figure();
        matplot::hold(matplot::on);

        // Couleur = classe et symbole = ensemble.
        matplot::plot(train_mines.indices, train_mines.distances, "bo");
        matplot::plot(train_rocks.indices, train_rocks.distances, "ro");
        matplot::plot(test_mines.indices, test_mines.distances, "bx");
        matplot::plot(test_rocks.indices, test_rocks.distances, "rx");

        // Hyperplan : distance signée nulle.
        matplot::plot(
            vector{0.0, static_cast<double>(position) + 1.0},
            vector{0.0, 0.0},
            "k--"
        );

        // Séparation visuelle entre train et test.
        matplot::plot(
            vector{separation, separation},
            vector{y_min, y_max},
            "k:"
        );

        matplot::legend({
                "Train : Mine (+1)",
                "Train : Roche (-1)",
                "Test : Mine (+1)",
                "Test : Roche (-1)",
                "Hyperplan : d = 0",
                "Separation train/test"
            }
        );

        matplot::xlabel("Exemples ordonnes par ensemble puis par classe");
        matplot::ylabel("Distance signee (w.x + b) / ||w||");
        matplot::title("Repartition des donnees SONAR par rapport a l'hyperplan");

        matplot::xlim({0.0, static_cast<double>(position) + 1.0});
        matplot::ylim({y_min, y_max});
        matplot::grid(matplot::on);

        matplot::hold(matplot::off);
        matplot::save(filename);
    }
}

void validate_epochs(const SonarDataset& sonar, const size_t epochs, const Perceptron& perceptron)
{
    if (epochs == static_cast<size_t>(-1))
        cout << "Limite de 10000 epoques atteinte sans convergence." << endl;
    else
        cout << "Convergence apres " << epochs << " epoques." << endl;
    cout << "Ea = " << 100.0 * error_rate(perceptron, sonar.train) << endl;
    cout << "Eg = " << 100.0 * error_rate(perceptron, sonar.test) << endl;
}

void experiment_early_stopping(const Dataset& complete)
{
    if (complete.size() != 208)
        throw invalid_argument("208 exemples attendus");

    vector<pair<vector<double>, int>> examples(
        complete.begin(), complete.end()
    );

    constexpr size_t repetitions = 20;
    constexpr size_t max_epochs = 10000;
    constexpr size_t patience = 50;

    mt19937 rng(42);
    uniform_real_distribution distribution(-1.0, 1.0);

    double sum_Ea = 0.0;
    double sum_Ev = 0.0;
    double sum_Et = 0.0;

    cout << fixed << setprecision(2);
    cout << endl << "Essai\tEpoque retenue\tEa (%)\tEv (%)\tEt (%)" << endl;

    for (size_t run = 1; run <= repetitions; ++run)
    {
        ranges::shuffle(examples, rng);

        const Dataset LA(examples.begin(), examples.begin() + 104);
        const Dataset LV(examples.begin() + 104, examples.begin() + 146);
        const Dataset LT(examples.begin() + 146, examples.end());

        vector<double> weights(examples.front().first.size());
        for (double& w : weights) w = distribution(rng);

        Perceptron model(weights, 0.6f, 1.0, LA);

        Perceptron best = model;
        double best_Ev = error_rate(model, LV);
        size_t best_epoch = 0;
        size_t without_improvement = 0;

        for (size_t epoch = 1; epoch <= max_epochs; ++epoch)
        {
            model.learn(Perceptron::ONLINE, 1);

            const double Ev = error_rate(model, LV);

            if (Ev < best_Ev)
            {
                best = model;
                best_Ev = Ev;
                best_epoch = epoch;
                without_improvement = 0;
            }
            else
            {
                ++without_improvement;
            }

            if (without_improvement >= patience || model.verify()) break;
        }

        const double Ea = 100.0 * error_rate(best, LA);
        const double Ev = 100.0 * error_rate(best, LV);
        const double Et = 100.0 * error_rate(best, LT);

        sum_Ea += Ea;
        sum_Ev += Ev;
        sum_Et += Et;

        cout << run << '\t'
            << best_epoch << '\t'
            << Ea << '\t'
            << Ev << '\t'
            << Et << endl;
    }

    cout << endl << "Moyenne sur " << repetitions << " essais :" << endl;
    cout << "Ea = " << sum_Ea / repetitions << endl;
    cout << "Ev = " << sum_Ev / repetitions << endl;
    cout << "Et = " << sum_Et / repetitions << endl;
}

int main()
{
    const SonarDataset sonar = get_dataset();

    const Dataset& learn_data = sonar.train;
    const Dataset& test_data = sonar.test;
    const Dataset& all_data = sonar.complete;

    // permet de vérifier que les datasets sont loaded correctement
    cout << "Train   : " << learn_data.size() << endl;
    cout << "Test    : " << test_data.size() << endl;
    cout << "Complet : " << all_data.size() << endl;

    // on train sur cpu pour l'instant car pas encore merge la branche de gpu

    Perceptron perceptron(60, 0.6, 1.0, learn_data); // dimension 60
    const auto epochs = perceptron.learn(Perceptron::ONLINE, 10000);

    validate_epochs(sonar, epochs, perceptron);

    // draw stability graph
    draw_stability(test_data, learn_data, perceptron, "normal.png");

    cout << "Inversion de learn et test" << endl;

    // all the same but reversed
    Perceptron perceptron2(60, 0.6, 1.0, test_data);
    const auto epochs2 = perceptron2.learn(Perceptron::ONLINE, 10000);
    validate_epochs(sonar, epochs2, perceptron2);
    draw_stability(learn_data, test_data, perceptron2, "reversed.png");

    // pocket algorithm
    pocket::Options options;
    options.max_epochs = 10000;
    options.max_errors = 5;
    options.seed = 42;
    options.eta = 0.6;

    for (const auto init : {pocket::Initialization::Random, pocket::Initialization::Hebb})
    {
        const auto result = pocket::fit(learn_data, init, options);

        const size_t errors_a = pocket::error_count(result.best, learn_data);
        const size_t errors_g = pocket::error_count(result.best, test_data);

        const double Ea = 100.0 * static_cast<double>(errors_a) / static_cast<double>(learn_data.size());
        const double Eg = 100.0 * static_cast<double>(errors_g) / static_cast<double>(test_data.size());

        cout << (init == pocket::Initialization::Random ? "Random" : "Hebb") << '\t'
            << errors_a << '/' << learn_data.size() << '\t'
            << Ea << '\t'
            << Eg << '\t'
            << result.epochs_started << '\t'
            << result.updates << '\t'
            << (result.target_reached ? "oui" : "non")
            << endl;
    }

    // part 3 learn on the whole set (learn + test)
    Perceptron perceptron_full(60, 0.6, 1.0, all_data); // dimension 60
    const auto epochs_full = perceptron_full.learn(Perceptron::ONLINE, 10000);

    if (epochs_full == static_cast<size_t>(-1))
        cout << "Limite de 10000 epoques atteinte sans convergence." << endl;

    else
        cout << "Convergence apres " << epochs_full << " epoques." << endl;
    cout << "Ea = " << 100.0 * error_rate(perceptron_full, all_data) << endl;

    // part 4
    experiment_early_stopping(all_data);
}
