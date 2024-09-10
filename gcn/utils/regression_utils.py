import numpy as np
from age_predictor import Age_Predictor, Age_Predictor_for_Node_Level_Measures, Structural_Age_Predictor
import matplotlib.pyplot as plt
import numpy as np
import os 

regression_type_strs = [
                        # "linear",
                        # "ridge",  
                        "random_forest", # NOTE: Random Forest time increases significantly when adding PLV measures
                        # "neural_network", 
                        # "huber", # NOTE: HUBER REGRESSOR ALSO TAKES FOREVER WHEN USING PLV and DOES TERRIBLY WITH LARGE NUMBER OF FEATURES
                        # "linear_svr", 
                        # "rbf_svr", # NOTE: RBF SVR behaves abnormally in regression (predicts a constant value throughout)
                        # "gaussian_process", 
                        # "decision_tree", # NOTE: DECISION TREE ALSO TAKES FOREVER WHEN USING PLV
                        # 'k_neighbors',
                        # 'ada_boost' # NOTE: ADA BOOST ALSO TAKES FOREVER WHEN USING PLV
                        ]

def compute_and_plot_performance_of_regressor_models(date, log_num):
    # regression_type_strs = [
    # "linear",
    # "ridge",  
    # "random_forest", 
    # "neural_network", 
    # "huber", 
    # "linear_svr", 
    # # "rbf_svr", # NOTE: RBF SVR behaves abnormally in regression (predicts a constant value throughout)
    # "gaussian_process", 
    # "decision_tree", 
    # 'k_neighbors',
    # 'ada_boost'
    # ]

    performance_matrix = np.zeros((len(regression_type_strs), 3))
    avg_mae_score = 0
    ridge_alpha = 100
    print(f"USING RIDGE ALPHA VALUE OF {ridge_alpha}")
    for index, regression_type_str in enumerate(regression_type_strs):        
        age_predictor_model = Age_Predictor(
                date, 
                log_num, 
                regression_type_str,
                alpha=ridge_alpha)
        
        mae, mse, corr, r2 = age_predictor_model.regression()
        avg_mae_score += mae
        
        performance_matrix[index, 0] += mae
        performance_matrix[index, 1] += mse
        performance_matrix[index, 2] += corr
    avg_mae_score /= len(regression_type_strs)
    
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title('Age Prediction Regressor Model Performance')
    # plt.title('FD Distance instead of Square Distance Age Prediction Regressor Model Performance')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score

def compute_and_plot_average_performance_of_regressor_models(date):
    # regression_type_strs = [
    # "linear",
    # "ridge",  
    # "random_forest", 
    # # "neural_network", 
    # "huber", 
    # "linear_svr", 
    # # "rbf_svr", # NOTE: RBF SVR behaves abnormally in regression (predicts a constant value throughout)
    # "gaussian_process", 
    # "decision_tree", 
    # 'k_neighbors',
    # 'ada_boost'
    # ]

    performance_matrix = np.zeros((len(regression_type_strs), 3))
    avg_mae_score = 0
    for index, regression_type_str in enumerate(regression_type_strs):
        run_numbers = os.listdir(os.path.join(os.getcwd(), "logs", "lp", date))
        print(run_numbers)
        run_mae_score = 0
        for run_number in run_numbers:
            age_predictor_model = Age_Predictor(
                    date, 
                    run_number, 
                    regression_type_str, 
                    use_viz = False)
            mae, mse, corr, r2 = age_predictor_model.regression()
            run_mae_score += mae
            performance_matrix[index, 0] += mae
            performance_matrix[index, 1] += mse
            performance_matrix[index, 2] += corr
        performance_matrix[index, 0] /= len(run_numbers)
        performance_matrix[index, 1] /= len(run_numbers)
        performance_matrix[index, 2] /= len(run_numbers)
        run_mae_score /= len(run_numbers)
        avg_mae_score += run_mae_score
    avg_mae_score /= len(regression_type_strs)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title('Average Age Prediction Regressor Model Performance')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score

def compute_and_plot_average_performance_of_regressor_models_for_benchmarking(date, regression_model_str, seed):
    if regression_model_str == "adaboost":
        regression_type_strs = [
            'ada_boost'
        ]
    elif regression_model_str == "ridge":
        regression_type_strs = [
            "ridge"
        ]
    elif regression_model_str == "rf":
        regression_type_strs = [
            "random_forest"
        ]
    np.random.seed(seed)
    performance_matrix = np.zeros((len(regression_type_strs), 3))
    avg_mae_score = 0
    for index, regression_type_str in enumerate(regression_type_strs):
        run_numbers = os.listdir(os.path.join(os.getcwd(), "logs", "lp", date))
        print(run_numbers)
        run_mae_score = 0
        for run_number in run_numbers:
            age_predictor_model = Age_Predictor(
                    date, 
                    run_number, 
                    regression_type_str, 
                    use_viz = False,
                    seed = seed)
            mae, mse, corr, r2 = age_predictor_model.regression()
            run_mae_score += mae
            performance_matrix[index, 0] += mae
            performance_matrix[index, 1] += mse
            performance_matrix[index, 2] += corr
        performance_matrix[index, 0] /= len(run_numbers)
        performance_matrix[index, 1] /= len(run_numbers)
        performance_matrix[index, 2] /= len(run_numbers)
        run_mae_score /= len(run_numbers)
        avg_mae_score += run_mae_score
    avg_mae_score /= len(regression_type_strs)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title('Average Age Prediction Regressor Model Performance')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score
def compute_and_plot_performance_of_regressor_models_for_node_level_measures(measure_str : str, date : str, log_num : str):
    # regression_type_strs = [
    # "linear",
    # "ridge",  
    # "random_forest", 
    # # "neural_network", 
    # "huber", 
    # "linear_svr", 
    # # "rbf_svr", # NOTE: RBF SVR behaves abnormally in regression (predicts a constant value throughout)
    # "gaussian_process", 
    # "decision_tree", 
    # 'k_neighbors',
    # 'ada_boost'
    # ]

    # if "plv" in measure_str:
    #     regression_type_strs = [
    #         "linear",
    #         "ridge",  
    #         # "random_forest", 
    #         # "neural_network", 
    #         "huber", 
    #         "linear_svr", 
    #         # "rbf_svr", # NOTE: RBF SVR behaves abnormally in regression (predicts a constant value throughout)
    #         "gaussian_process", 
    #         "decision_tree", 
    #         'k_neighbors',
    #         # 'ada_boost'
    #     ]

    performance_matrix = np.zeros((len(regression_type_strs), 3))
    avg_mae_score = 0

    for index, regression_type_str in enumerate(regression_type_strs):        
        print(regression_type_str, "REGRESSION TYPE")
        age_predictor_model = Age_Predictor_for_Node_Level_Measures(measure_str,
            date, 
            log_num,
            regression_type_str,
            use_viz=False)
        mae, mse, corr, r2 = age_predictor_model.regression()
        avg_mae_score += mae

        performance_matrix[index, 0] += mae
        performance_matrix[index, 1] += mse
        performance_matrix[index, 2] += corr
    avg_mae_score /= len(regression_type_strs)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title(f'Age Prediction Regressor Model Performance : {measure_str}')
    
    # plt.title('FD Distance instead of Square Distance Age Prediction Regressor Model Performance')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score


def compute_and_plot_average_performance_of_regressor_models_for_node_level_measures(measure_str : str, date : str):
    # regression_type_strs = [
    # "linear",
    # "ridge",
    # "random_forest", 
    # "neural_network", 
    # "huber", 
    # "linear_svr", 
    # # "rbf_svr", # NOTE: RBF SVR behaves abnormally in regression (predicts a constant value throughout)
    # "gaussian_process", 
    # "decision_tree", 
    # 'k_neighbors',
    # 'ada_boost'
    # ]

    performance_matrix = np.zeros((len(regression_type_strs), 3))
    avg_mae_score = 0
    run_numbers = os.listdir(os.path.join(os.getcwd(), "logs", "lp", date))
    run_numbers.sort()
    run_mae_scores = list()
    for run_number in run_numbers:
        run_mae_score = 0
        for index, regression_type_str in enumerate(regression_type_strs):
            age_predictor_model = Age_Predictor_for_Node_Level_Measures(
                    measure_str,
                    date, 
                    run_number, 
                    regression_type_str, 
                    use_viz=False)
            mae, mse, corr, r2 = age_predictor_model.regression()
            run_mae_score += mae
            performance_matrix[index, 0] += mae
            performance_matrix[index, 1] += mse
            performance_matrix[index, 2] += corr
        performance_matrix[index, 0] /= len(regression_type_strs)
        performance_matrix[index, 1] /= len(regression_type_strs)
        performance_matrix[index, 2] /= len(regression_type_strs)
        run_mae_score /= len(regression_type_strs)
        avg_mae_score += run_mae_score
        run_mae_scores.append(run_mae_score)
    avg_mae_score /= len(run_numbers)
    performance_matrix[index, 0] /= len(run_numbers)
    performance_matrix[index, 1] /= len(run_numbers)
    performance_matrix[index, 2] /= len(run_numbers)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title(f'Average Regressor Model Performance : {measure_str}')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score, run_mae_scores

def compute_and_plot_average_performance_of_regressor_models_for_node_level_measures_for_benchmarking(measure_str : str, 
                                                                                                    date : str, 
                                                                                                    regression_model_str : str, 
                                                                                                    seed : int):
    if regression_model_str == "adaboost":
        regression_type_strs = [
            'ada_boost'
        ]
    elif regression_model_str == "ridge":
        regression_type_strs = [
            "ridge"
        ]
    elif regression_model_str == "rf":
        regression_type_strs = [
            "random_forest"
        ]

    np.random.seed(seed)

    performance_matrix = np.zeros((len(regression_type_strs), 3))
    avg_mae_score = 0
    run_numbers = os.listdir(os.path.join(os.getcwd(), "logs", "lp", date))
    run_numbers.sort()
    run_mae_scores = list()
    for run_number in run_numbers:
        run_mae_score = 0
        for index, regression_type_str in enumerate(regression_type_strs):
            age_predictor_model = Age_Predictor_for_Node_Level_Measures(
                    measure_str,
                    date, 
                    run_number, 
                    regression_type_str, 
                    use_viz=False,
                    seed=seed)
            mae, mse, corr, r2 = age_predictor_model.regression()
            run_mae_score += mae
            performance_matrix[index, 0] += mae
            performance_matrix[index, 1] += mse
            performance_matrix[index, 2] += corr
        performance_matrix[index, 0] /= len(regression_type_strs)
        performance_matrix[index, 1] /= len(regression_type_strs)
        performance_matrix[index, 2] /= len(regression_type_strs)
        run_mae_score /= len(regression_type_strs)
        avg_mae_score += run_mae_score
        run_mae_scores.append(run_mae_score)
    avg_mae_score /= len(run_numbers)
    performance_matrix[index, 0] /= len(run_numbers)
    performance_matrix[index, 1] /= len(run_numbers)
    performance_matrix[index, 2] /= len(run_numbers)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title(f'Average Regressor Model Performance : {measure_str}')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score, run_mae_scores

def compute_and_plot_performance_of_structural_regressor_models(measure_str : str, date : str, log_num : str):

    ridge_alpha = 100
    
    print(f"USING RIDGE ALPHA VALUE OF {ridge_alpha}")
    avg_mae_score = 0
    performance_matrix = np.zeros((len(regression_type_strs), 3))
    for index, regression_type_str in enumerate(regression_type_strs):        
        age_predictor_model = Structural_Age_Predictor(measure_str,
            date, 
            log_num,
            regression_type_str,
            alpha=ridge_alpha)
        mae, mse, corr, r2 = age_predictor_model.regression()
        performance_matrix[index, 0] += mae
        performance_matrix[index, 1] += mse
        performance_matrix[index, 2] += corr
        avg_mae_score += mae
    avg_mae_score /= len(regression_type_strs)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title(f'Age Prediction Regressor Model Performance : {measure_str}')
    
    # plt.title('FD Distance instead of Square Distance Age Prediction Regressor Model Performance')
    plt.show()

    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score

def compute_and_plot_average_performance_of_structural_regressor_models(measure_str : str, date : str):
    # regression_type_strs = [
    # "linear",
    # "ridge",
    # "random_forest", 
    # "neural_network", 
    # "huber", 
    # "linear_svr", 
    # # "rbf_svr", # NOTE: RBF SVR behaves abnormally in regression (predicts a constant value throughout)
    # "gaussian_process", 
    # "decision_tree", 
    # 'k_neighbors',
    # 'ada_boost'
    # ]

    performance_matrix = np.zeros((len(regression_type_strs), 3))
    avg_mae_score = 0
    run_numbers = os.listdir(os.path.join(os.getcwd(), "logs", "lp", date))
    run_numbers.sort()
    run_mae_scores = list()
    model_to_mae_scores = dict()
    for run_number in run_numbers:
        run_mae_score = 0
        for index, regression_type_str in enumerate(regression_type_strs):
            age_predictor_model = Structural_Age_Predictor(
                    measure_str,
                    date, 
                    run_number, 
                    regression_type_str)
            mae, mse, corr, r2 = age_predictor_model.regression()
            run_mae_score += mae
            performance_matrix[index, 0] += mae
            performance_matrix[index, 1] += mse
            performance_matrix[index, 2] += corr
            model_to_mae_scores[regression_type_str] = model_to_mae_scores.get(regression_type_str, []) + [mae]

        performance_matrix[index, 0] /= len(regression_type_strs)
        performance_matrix[index, 1] /= len(regression_type_strs)
        performance_matrix[index, 2] /= len(regression_type_strs)
        run_mae_score /= len(regression_type_strs)
        avg_mae_score += run_mae_score
        run_mae_scores.append(run_mae_score)
    
    performance_matrix[index, 0] /= len(run_numbers)
    performance_matrix[index, 1] /= len(run_numbers)
    performance_matrix[index, 2] /= len(run_numbers)
    avg_mae_score /= len(run_numbers)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title(f'Average Regressor Model Performance : {measure_str}')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score, run_mae_scores, model_to_mae_scores

def compute_and_plot_average_performance_of_structural_regressor_models_for_benchmarking(measure_str : str, 
                                                                                        date : str, 
                                                                                        regression_model_str : str, 
                                                                                        seed : int):
    if regression_model_str == "adaboost":
        regression_type_strs = [
            'ada_boost'
        ]
    elif regression_model_str == "ridge":
        regression_type_strs = [
            "ridge"
        ]
    elif regression_model_str == "rf":
        regression_type_strs = [
            "random_forest"
        ]
    np.random.seed(seed)
    performance_matrix = np.zeros((len(regression_type_strs), 3))
    avg_mae_score = 0
    run_numbers = os.listdir(os.path.join(os.getcwd(), "logs", "lp", date))
    run_numbers.sort()
    run_mae_scores = list()
    model_to_mae_scores = dict()
    for run_number in run_numbers:
        run_mae_score = 0
        for index, regression_type_str in enumerate(regression_type_strs):
            age_predictor_model = Structural_Age_Predictor(
                    measure_str,
                    date, 
                    run_number, 
                    regression_type_str)
            mae, mse, corr, r2 = age_predictor_model.regression()
            run_mae_score += mae
            performance_matrix[index, 0] += mae
            performance_matrix[index, 1] += mse
            performance_matrix[index, 2] += corr
            model_to_mae_scores[regression_type_str] = model_to_mae_scores.get(regression_type_str, []) + [mae]

        performance_matrix[index, 0] /= len(regression_type_strs)
        performance_matrix[index, 1] /= len(regression_type_strs)
        performance_matrix[index, 2] /= len(regression_type_strs)
        run_mae_score /= len(regression_type_strs)
        avg_mae_score += run_mae_score
        run_mae_scores.append(run_mae_score)
    
    performance_matrix[index, 0] /= len(run_numbers)
    performance_matrix[index, 1] /= len(run_numbers)
    performance_matrix[index, 2] /= len(run_numbers)
    avg_mae_score /= len(run_numbers)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(len(regression_type_strs)):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title(f'Average Regressor Model Performance : {measure_str}')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score, run_mae_scores, model_to_mae_scores



def compute_and_plot_average_performance_of_structural_regressor_model(measure_str : str, date : str, model_str, use_viz=False):
    # regression_type_strs = [
    # "linear",
    # "ridge",
    # "random_forest", 
    # "neural_network", 
    # "huber", 
    # "linear_svr", 
    # # "rbf_svr", # NOTE: RBF SVR behaves abnormally in regression (predicts a constant value throughout)
    # "gaussian_process", 
    # "decision_tree", 
    # 'k_neighbors',
    # 'ada_boost'
    # ]
    if model_str not in regression_type_strs: raise AssertionError(f"Invalid model str : {model_str}")
    performance_matrix = np.zeros((1, 3))
    avg_mae_score = 0
    run_numbers = os.listdir(os.path.join(os.getcwd(), "logs", "lp", date))
    run_mae_scores = list()
    for run_number in run_numbers:
        run_mae_score = 0
        
        age_predictor_model = Structural_Age_Predictor(
                measure_str,
                date, 
                run_number, 
                model_str, 
                use_viz=use_viz)
        mae, mse, corr, r2 = age_predictor_model.regression()
        run_mae_score += mae
        performance_matrix[0, 0] += mae
        performance_matrix[0, 1] += mse
        performance_matrix[0, 2] += corr

        run_mae_score /= len(regression_type_strs)
        avg_mae_score += run_mae_score
        run_mae_scores.append(run_mae_score)
    
    performance_matrix[0, 0] /= len(run_numbers)
    performance_matrix[0, 1] /= len(run_numbers)
    performance_matrix[0, 2] /= len(run_numbers)

    avg_mae_score /= len(run_numbers)
    plt.imshow(performance_matrix, cmap='Blues', interpolation='None')

    # Add entry numbers on top of the colors
    for i in range(1):
        for j in range(len(performance_matrix[0])):
            plt.text(j, i, f'{performance_matrix[i][j]:.2f}', ha='center', va='center', color='black', size='xx-small')

    plt.colorbar(label='Values')
    plt.xticks(np.arange(len(performance_matrix[0])), ["mae", "mse", "correlation"], rotation = 90)
    plt.yticks(np.arange(len(regression_type_strs)), regression_type_strs)
    plt.title(f'Average Regressor Model Performance : {measure_str}')
    plt.show()
    print(f"Average Regressor Model MAE Score: {avg_mae_score}")
    return avg_mae_score, run_mae_scores






