#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <random>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <chrono>  // For time calculations

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define BOLD "\033[1m"

// Mutex for thread safety
std::mutex mtx;

// Utility for checking valid input
bool isValidInput() {
    if (std::cin.fail()) {
        std::cin.clear(); // clear the error flag
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // discard bad input
        std::cout << RED << "Invalid input. Please try again.\n" << RESET;
        return false;
    }
    return true;
}

// Utility for logging results and inputs to a file
void logResults(const std::string& taskName, const std::string& result, const std::string& inputs, long long duration) {
    std::ofstream logfile("computation_results.log", std::ios::app);
    if (logfile.is_open()) {
        logfile << "Task: " << taskName << "\n";
        logfile << "Inputs: " << inputs << "\n";
        logfile << "Result: " << result << "\n";
        logfile << "Execution Time: " << duration << " milliseconds\n";
        logfile << "-----------------------\n";
        logfile.close();
    }
}

// Function for solving linear equations using Gaussian elimination (simplified)
void gaussianElimination(std::vector<std::vector<double>>& matrix, std::vector<double>& result, long long duration) {
    int n = matrix.size();
    for (int i = 0; i < n; i++) {
        if (matrix[i][i] == 0) {
            std::cerr << RED << "Error: Singular matrix. Gaussian elimination cannot proceed.\n" << RESET;
            return;
        }
        for (int k = i + 1; k < n; k++) {
            double factor = matrix[k][i] / matrix[i][i];
            for (int j = i; j < n; j++) {
                matrix[k][j] -= factor * matrix[i][j];
            }
            result[k] -= factor * result[i];
        }
    }

    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = result[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= matrix[i][j] * x[j];
        }
        x[i] = x[i] / matrix[i][i];
    }

    std::lock_guard<std::mutex> lock(mtx);
    std::cout << GREEN << "Solution to the system of equations:\n";
    std::string solution;
    for (double xi : x) {
        std::cout << CYAN << xi << " ";
        solution += std::to_string(xi) + " ";
    }
    std::cout << RESET << std::endl;

    std::string inputs = "Matrix size: " + std::to_string(n);
    for (int i = 0; i < n; ++i) {
        inputs += "\nRow " + std::to_string(i) + ": ";
        for (int j = 0; j < n; ++j) {
            inputs += std::to_string(matrix[i][j]) + " ";
        }
        inputs += "\nResult: " + std::to_string(result[i]);
    }

    logResults("Gaussian Elimination", solution, inputs, duration);
}

// Polynomial evaluation using Horner's method
double hornerMethod(const std::vector<double>& coeffs, double x) {
    double result = coeffs[0];
    for (size_t i = 1; i < coeffs.size(); i++) {
        result = result * x + coeffs[i];
    }
    return result;
}

// Parallel Polynomial Evaluation using std::async
void parallelPolynomialEvaluationAsync(const std::vector<double>& coeffs, const std::vector<double>& points, long long duration) {
    std::vector<std::future<double>> futures;
    std::vector<double> results(points.size());
    std::string polyResult = "";

    for (size_t i = 0; i < points.size(); ++i) {
        futures.push_back(std::async(std::launch::async, [coeffs](double point) {
            return hornerMethod(coeffs, point);
        }, points[i]));
    }

    for (size_t i = 0; i < points.size(); ++i) {
        results[i] = futures[i].get();
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << GREEN << "Polynomial evaluated at x = " << YELLOW << points[i] << GREEN << ": " << CYAN << results[i] << RESET << std::endl;
        polyResult += "x=" + std::to_string(points[i]) + ": " + std::to_string(results[i]) + "\n";
    }

    std::string inputs = "Coefficients: ";
    for (const auto& coeff : coeffs) {
        inputs += std::to_string(coeff) + " ";
    }
    inputs += "\nEvaluation points: ";
    for (const auto& point : points) {
        inputs += std::to_string(point) + " ";
    }

    logResults("Polynomial Evaluation", polyResult, inputs, duration);
}

// Numerical integration using the Trapezoidal rule
double trapezoidalRule(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double result = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; ++i) {
        result += f(a + i * h);
    }
    return result * h;
}

// Parallel numerical integration
void parallelIntegration(double (*f)(double), double a, double b, int n, int numThreads, long long duration) {
    std::vector<std::thread> threads;
    double step = (b - a) / numThreads;
    double result = 0.0;
    std::mutex resultMutex;

    for (int i = 0; i < numThreads; ++i) {
        threads.push_back(std::thread([&result, f, a, step, i, n, numThreads, &resultMutex]() {
            double partialResult = trapezoidalRule(f, a + i * step, a + (i + 1) * step, n / numThreads);
            std::lock_guard<std::mutex> lock(resultMutex);
            result += partialResult;
        }));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << GREEN << "Result of numerical integration: " << CYAN << result << RESET << std::endl;

    std::string inputs = "a: " + std::to_string(a) + ", b: " + std::to_string(b) + ", sub-intervals: " + std::to_string(n);
    logResults("Numerical Integration", std::to_string(result), inputs, duration);
}

// Monte Carlo simulation for Pi calculation
void monteCarloPi(int numPoints, int numThreads, long long duration) {
    std::vector<std::thread> threads;
    std::vector<int> insideCircle(numThreads, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int t = 0; t < numThreads; ++t) {
        threads.push_back(std::thread([&insideCircle, &gen, &dis, numPoints, t, numThreads]() {
            for (int i = 0; i < numPoints / numThreads; ++i) {
                double x = dis(gen);
                double y = dis(gen);
                if (x * x + y * y <= 1.0) {
                    insideCircle[t]++;
                }
            }
        }));
    }

    for (auto& t : threads) {
        t.join();
    }

    int totalInside = 0;
    for (int count : insideCircle) {
        totalInside += count;
    }

    double pi = 4.0 * totalInside / numPoints;
    std::cout << GREEN << "Estimated value of Pi: " << CYAN << pi << RESET << std::endl;

    std::string inputs = "Number of points: " + std::to_string(numPoints) + ", Number of threads: " + std::to_string(numThreads);
    logResults("Monte Carlo Pi Estimation", std::to_string(pi), inputs, duration);
}

// Newton-Raphson method for finding roots
double newtonRaphson(double (*f)(double), double (*df)(double), double x0, int maxIter, double tol) {
    double x = x0;
    for (int i = 0; i < maxIter; ++i) {
        double fx = f(x);
        if (std::abs(fx) < tol) {
            return x;
        }
        x = x - fx / df(x);
    }
    return x;
}

// Parallel non-linear solver using Newton-Raphson
void parallelNonLinearSolver(double (*f)(double), double (*df)(double), const std::vector<double>& guesses, long long duration) {
    std::vector<std::thread> threads;
    std::string rootResults = "";
    for (double guess : guesses) {
        threads.push_back(std::thread([f, df, guess, &rootResults]() {
            double root = newtonRaphson(f, df, guess, 100, 1e-6);
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << GREEN << "Root found near initial guess " << YELLOW << guess << GREEN << ": " << CYAN << root << RESET << std::endl;
            rootResults += "Initial guess: " + std::to_string(guess) + ", Root: " + std::to_string(root) + "\n";
        }));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::string inputs = "Initial guesses: ";
    for (const auto& guess : guesses) {
        inputs += std::to_string(guess) + " ";
    }

    logResults("Newton-Raphson Root Finding", rootResults, inputs, duration);
}

// Function to display the menu and get user choice
int displayMenu() {
    int choice;
    std::cout << BOLD << BLUE << "\n===== Main Menu =====\n" << RESET;
    std::cout << CYAN << "1. " << RESET << "Solve a system of linear equations (Gaussian elimination)\n";
    std::cout << CYAN << "2. " << RESET << "Evaluate polynomial at multiple points (Horner's method)\n";
    std::cout << CYAN << "3. " << RESET << "Compute numerical integration (Trapezoidal rule)\n";
    std::cout << CYAN << "4. " << RESET << "Estimate Pi using Monte Carlo simulation\n";
    std::cout << CYAN << "5. " << RESET << "Solve non-linear equations (Newton-Raphson)\n";
    std::cout << CYAN << "6. " << RESET << "View computation history\n";
    std::cout << CYAN << "7. " << RESET << "Run multiple algorithms concurrently\n";  // New Option 7
    std::cout << CYAN << "8. " << RESET << "Exit\n";  // Exit now option 8
    std::cout << BOLD << YELLOW << "Enter your choice (1-8): " << RESET;
    std::cin >> choice;

    while (!isValidInput() || (choice < 1 || choice > 8)) {
        std::cout << RED << "Invalid choice. Please enter a number between 1 and 8: " << RESET;
        std::cin >> choice;
    }

    return choice;
}

// Function to print history from the log file
void printHistory() {
    std::ifstream logfile("computation_results.log");
    if (!logfile.is_open()) {
        std::cerr << RED << "Error: Could not open log file." << RESET << std::endl;
        return;
    }

    std::string line;
    std::cout << BOLD << MAGENTA << "\n===== Computation History =====\n" << RESET;
    while (getline(logfile, line)) {
        std::cout << CYAN << line << RESET << std::endl;
    }
    logfile.close();
}

// Input matrix for Gaussian elimination
void inputMatrix(std::vector<std::vector<double>>& matrix, std::vector<double>& result) {
    int n;
    std::cout << YELLOW << "Enter the number of equations (matrix size): " << RESET;
    std::cin >> n;
    while (!isValidInput()) {
        std::cout << RED << "Enter a valid number for the size of the matrix: " << RESET;
        std::cin >> n;
    }
    matrix.resize(n, std::vector<double>(n));
    result.resize(n);

    std::cout << MAGENTA << "Example: If you have 2 equations with 2 variables, enter 2 1 1, 3 4 for coefficients and 1 5 for the result.\n" << RESET;
    std::cout << YELLOW << "Enter the coefficients of the matrix row by row:\n" << RESET;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> matrix[i][j];
            while (!isValidInput()) {
                std::cout << RED << "Please enter a valid number: " << RESET;
                std::cin >> matrix[i][j];
            }
        }
    }

    std::cout << YELLOW << "Enter the results vector:\n" << RESET;
    for (int i = 0; i < n; i++) {
        std::cin >> result[i];
        while (!isValidInput()) {
            std::cout << RED << "Please enter a valid number: " << RESET;
            std::cin >> result[i];
        }
    }
}

// Input for polynomial coefficients and evaluation points
void inputPolynomial(std::vector<double>& coeffs, std::vector<double>& points) {
    int degree, numPoints;
    std::cout << YELLOW << "Enter the degree of the polynomial: " << RESET;
    std::cin >> degree;
    while (!isValidInput()) {
        std::cout << RED << "Enter a valid degree: " << RESET;
        std::cin >> degree;
    }
    coeffs.resize(degree + 1);

    std::cout << MAGENTA << "Example: For a 2nd degree polynomial, enter coefficients like 3 2 1 (for 3x^2 + 2x + 1).\n" << RESET;
    std::cout << YELLOW << "Enter the coefficients of the polynomial (highest to lowest degree):\n" << RESET;
    for (int i = 0; i <= degree; ++i) {
        std::cin >> coeffs[i];
        while (!isValidInput()) {
            std::cout << RED << "Please enter a valid coefficient: " << RESET;
            std::cin >> coeffs[i];
        }
    }

    std::cout << YELLOW << "Enter the number of points at which to evaluate the polynomial: " << RESET;
    std::cin >> numPoints;
    while (!isValidInput()) {
        std::cout << RED << "Enter a valid number: " << RESET;
        std::cin >> numPoints;
    }
    points.resize(numPoints);

    std::cout << MAGENTA << "Example: Enter points like 1 2 3 to evaluate the polynomial at x = 1, 2, and 3.\n" << RESET;
    std::cout << YELLOW << "Enter the points:\n" << RESET;
    for (int i = 0; i < numPoints; ++i) {
        std::cin >> points[i];
        while (!isValidInput()) {
            std::cout << RED << "Please enter a valid point: " << RESET;
            std::cin >> points[i];
        }
    }
}

// Input for numerical integration
void inputIntegration(double& a, double& b, int& n) {
    std::cout << YELLOW << "Enter the lower limit of integration: " << RESET;
    std::cin >> a;
    while (!isValidInput()) {
        std::cout << RED << "Enter a valid number: " << RESET;
        std::cin >> a;
    }

    std::cout << YELLOW << "Enter the upper limit of integration: " << RESET;
    std::cin >> b;
    while (!isValidInput()) {
        std::cout << RED << "Enter a valid number: " << RESET;
        std::cin >> b;
    }

    std::cout << YELLOW << "Enter the number of sub-intervals: " << RESET;
    std::cin >> n;
    while (!isValidInput()) {
        std::cout << RED << "Enter a valid number: " << RESET;
        std::cin >> n;
    }
}

// Input for non-linear equation solver
void inputNonLinearSolver(std::vector<double>& guesses) {
    int numGuesses;
    std::cout << YELLOW << "Enter the number of initial guesses: " << RESET;
    std::cin >> numGuesses;
    while (!isValidInput()) {
        std::cout << RED << "Enter a valid number: " << RESET;
        std::cin >> numGuesses;
    }
    guesses.resize(numGuesses);

    std::cout << MAGENTA << "Example: Enter guesses like 1 2 3 to provide different starting points for root finding.\n" << RESET;
    std::cout << YELLOW << "Enter the initial guesses:\n" << RESET;
    for (int i = 0; i < numGuesses; ++i) {
        std::cin >> guesses[i];
        while (!isValidInput()) {
            std::cout << RED << "Please enter a valid guess: " << RESET;
            std::cin >> guesses[i];
        }
    }
}

// Test function for integration and root finding (f(x) = x^2 - 2)
double testFunction(double x) {
    return x * x - 2;
}

// Derivative of test function (f'(x) = 2x)
double testFunctionDerivative(double x) {
    return 2 * x;
}

// Function to run multiple algorithms concurrently
void runMultipleAlgorithms(int numAlgorithms) {
    std::vector<int> choices(numAlgorithms);

    // Input the choices for algorithms
    for (int i = 0; i < numAlgorithms; ++i) {
        std::cout << YELLOW << "Enter the algorithm number (1-5) for algorithm " << i + 1 << ": " << RESET;
        std::cin >> choices[i];
        while (!isValidInput() || (choices[i] < 1 || choices[i] > 5)) {
            std::cout << RED << "Invalid choice. Please enter a number between 1 and 5: " << RESET;
            std::cin >> choices[i];
        }
    }

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear input buffer after entering values

    std::vector<std::function<void()>> tasks;
    std::vector<std::function<void()>> inputsCollectionTasks;

    // Collect all inputs and prepare tasks
    std::string overallLog = "";
    for (int choice : choices) {
        if (choice == 1) {
            // Collect inputs first, capture tasks by reference
            inputsCollectionTasks.push_back([&tasks, &overallLog]() {
                std::vector<std::vector<double>> matrix;
                std::vector<double> result;
                inputMatrix(matrix, result);
                tasks.push_back([&matrix, &result, &overallLog]() {  // Capture the inputs by reference for this task
                    auto start = std::chrono::high_resolution_clock::now();
                    gaussianElimination(matrix, result, 0);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << BOLD << BLUE << "Time for Gaussian Elimination: " << YELLOW << duration << " milliseconds\n" << RESET;
                    overallLog += "Gaussian Elimination completed in " + std::to_string(duration) + " ms\n";
                });
            });
        } else if (choice == 2) {
            // Collect inputs for polynomial evaluation
            inputsCollectionTasks.push_back([&tasks, &overallLog]() {
                std::vector<double> coeffs;
                std::vector<double> points;
                inputPolynomial(coeffs, points);
                tasks.push_back([=, &overallLog]() {  // Capture the inputs for this task
                    auto start = std::chrono::high_resolution_clock::now();
                    parallelPolynomialEvaluationAsync(coeffs, points, 0);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << BOLD << BLUE << "Time for Polynomial Evaluation: " << YELLOW << duration << " milliseconds\n" << RESET;
                    overallLog += "Polynomial Evaluation completed in " + std::to_string(duration) + " ms\n";
                });
            });
        } else if (choice == 3) {
            // Collect inputs for numerical integration
            inputsCollectionTasks.push_back([&tasks, &overallLog]() {
                double a, b;
                int n;
                inputIntegration(a, b, n);
                tasks.push_back([=, &overallLog]() {  // Capture the inputs for this task
                    auto start = std::chrono::high_resolution_clock::now();
                    parallelIntegration(testFunction, a, b, n, 4, 0);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << BOLD << BLUE << "Time for Numerical Integration: " << YELLOW << duration << " milliseconds\n" << RESET;
                    overallLog += "Numerical Integration completed in " + std::to_string(duration) + " ms\n";
                });
            });
        } else if (choice == 4) {
            // Collect inputs for Monte Carlo Pi estimation
            inputsCollectionTasks.push_back([&tasks, &overallLog]() {
                int numPoints;
                std::cout << YELLOW << "Enter the number of points to simulate (example: 100000): " << RESET;
                std::cin >> numPoints;
                tasks.push_back([=, &overallLog]() {  // Capture the inputs for this task
                    auto start = std::chrono::high_resolution_clock::now();
                    monteCarloPi(numPoints, 4, 0);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << BOLD << BLUE << "Time for Monte Carlo Pi Estimation: " << YELLOW << duration << " milliseconds\n" << RESET;
                    overallLog += "Monte Carlo Pi Estimation completed in " + std::to_string(duration) + " ms\n";
                });
            });
        } else if (choice == 5) {
            // Collect inputs for non-linear solver
            inputsCollectionTasks.push_back([&tasks, &overallLog]() {
                std::vector<double> guesses;
                inputNonLinearSolver(guesses);
                tasks.push_back([=, &overallLog]() {  // Capture the inputs for this task
                    auto start = std::chrono::high_resolution_clock::now();
                    parallelNonLinearSolver(testFunction, testFunctionDerivative, guesses, 0);
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    std::cout << BOLD << BLUE << "Time for Non-linear Solver: " << YELLOW << duration << " milliseconds\n" << RESET;
                    overallLog += "Non-linear Solver completed in " + std::to_string(duration) + " ms\n";
                });
            });
        }
    }

    // Collect inputs for all algorithms first
    for (auto& inputTask : inputsCollectionTasks) {
        inputTask();
    }

    // Execute all tasks concurrently after all inputs are collected
    std::vector<std::thread> threads;
    for (auto& task : tasks) {
        threads.emplace_back(task);
    }

    // Display task progress status
    for (int i = 0; i < threads.size(); ++i) {
        std::cout << GREEN << "Running Task " << i + 1 << "/" << threads.size() << "..." << RESET << std::endl;
    }

    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }

    std::cout << GREEN << "All tasks completed successfully!" << RESET << std::endl;

    // Log the overall progress to file
    logResults("Concurrent Algorithm Execution", "Multiple algorithms run", overallLog, 0);

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear input buffer after concurrent execution
}

// Main function to handle user interaction and execute tasks
int main() {
    int choice = displayMenu();

    while (choice != 8) {  // Exit now on choice 8
        if (choice == 1) {
            std::vector<std::vector<double>> matrix;
            std::vector<double> result;
            inputMatrix(matrix, result);

            auto start = std::chrono::high_resolution_clock::now();
            gaussianElimination(matrix, result, 0);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << BOLD << BLUE << "Time for Gaussian Elimination: " << YELLOW << duration << " milliseconds\n" << RESET;

        } else if (choice == 2) {
            std::vector<double> coeffs;
            std::vector<double> points;
            inputPolynomial(coeffs, points);

            auto start = std::chrono::high_resolution_clock::now();
            parallelPolynomialEvaluationAsync(coeffs, points, 0);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << BOLD << BLUE << "Time for Polynomial Evaluation: " << YELLOW << duration << " milliseconds\n" << RESET;

        } else if (choice == 3) {
            double a, b;
            int n;
            inputIntegration(a, b, n);

            auto start = std::chrono::high_resolution_clock::now();
            parallelIntegration(testFunction, a, b, n, 4, 0);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << BOLD << BLUE << "Time for Numerical Integration: " << YELLOW << duration << " milliseconds\n" << RESET;

        } else if (choice == 4) {
            int numPoints;
            std::cout << YELLOW << "Enter the number of points to simulate (example: 100000): " << RESET;
            std::cin >> numPoints;

            auto start = std::chrono::high_resolution_clock::now();
            monteCarloPi(numPoints, 4, 0);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << BOLD << BLUE << "Time for Monte Carlo Pi Estimation: " << YELLOW << duration << " milliseconds\n" << RESET;

        } else if (choice == 5) {
            std::vector<double> guesses;
            inputNonLinearSolver(guesses);

            auto start = std::chrono::high_resolution_clock::now();
            parallelNonLinearSolver(testFunction, testFunctionDerivative, guesses, 0);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << BOLD << BLUE << "Time for Non-linear Solver: " << YELLOW << duration << " milliseconds\n" << RESET;

        } else if (choice == 6) {
            // Option to view computation history
            printHistory();

        } else if (choice == 7) {
            // New option to run multiple algorithms concurrently
            int numAlgorithms;
            std::cout << YELLOW << "How many algorithms would you like to run concurrently? " << RESET;
            std::cin >> numAlgorithms;
            runMultipleAlgorithms(numAlgorithms);

        } else {
            std::cout << RED << "Invalid choice." << std::endl << RESET;
        }

        choice = displayMenu();
    }

    std::cout << GREEN << "Exiting program. Thank you for using the system!" << RESET << std::endl;
    return 0;
}

