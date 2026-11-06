#include <iostream>

#include "dataset.h"
#include "perceptron.h"

using namespace std;

int main()
{
    const SonarDataset sonar = get_dataset();

    const Dataset& learn_data = sonar.train;
    const Dataset& test_data = sonar.test;
    const Dataset& all_data = sonar.complete;

    // permet de vérifier que les datasets sont loaded correctement
    cout << "Train   : " << learn_data.size() << '\n';
    cout << "Test    : " << test_data.size() << '\n';
    cout << "Complet : " << all_data.size() << '\n';

    // on train sur cpu pour l'instant car pas encore merge la branche de gpu

    Perceptron perceptron(60, 0.5, 0.0, learn_data); // dimension 60
    cout << perceptron.learn(Perceptron::ONLINE);
}
