import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
# from cluster_shap import FindShap
import pandas as pd

import catboost
import sklearn.model_selection
import matplotlib.pyplot as plt
import matplotlib.colorbar
import shap
import numpy as np
import json


class RegressionModel:
    def __init__(self, 
            data : pd.DataFrame, 
            label : list[float],
            categorical_feature_indices : list[int]
        ):
        assert len(data.index) == len(label)
        self.data = data
        self.label = label
        self.random_state = 123
        self.categorical_feature_indices = categorical_feature_indices

        self.train_model()
        return

    @property
    def model(self) -> catboost.core.CatBoostRegressor:
        return self._model
    
    def train_model(self):
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(
                self.data, 
                self.label,
                test_size = 0.4,
                random_state = self.random_state
            )
        X_test, X_val, y_test, y_val = \
            sklearn.model_selection.train_test_split(
                X_test,
                y_test,
                test_size = 0.5,
                random_state = self.random_state
            )
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        
        dataset_train = catboost.Pool(X_train, y_train, 
            cat_features=self.categorical_feature_indices)
        dataset_test = catboost.Pool(X_test, y_test,
            cat_features=self.categorical_feature_indices)
        dataset_validate = catboost.Pool(X_val, y_val,
            cat_features=self.categorical_feature_indices)
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_validate = dataset_validate

        self.loss_function = "RMSE"
        loss_function = self.loss_function
        model = catboost.CatBoostRegressor(
            iterations= 100,
            learning_rate = 0.1,
            random_seed= 123,
            loss_function= loss_function,
            cat_features = self.categorical_feature_indices
        )

        # model.fit(dataset_train, verbose=False)
        # pred = model.predict(dataset_test)

        # Cross Validate
        self.cv_params = catboost.cv(
            pool = dataset_test,
            params = model.get_params(),
            seed = self.random_state,
            verbose_eval = False,
            fold_count= 10
        )
        cv_params = self.cv_params

        optimal_iterations = cv_params['test-%s-mean' % loss_function]\
            .idxmin()
        print("Optimal iterations:", optimal_iterations)

        model.set_params(
            iterations=100,
            random_seed = self.random_state
        )
        model.fit(dataset_train, verbose=False, plot=False)

        # for key, val in model.get_all_params().items():
        #     print("%20s" % key, val)
        

        
        
    
        self._model = model    
        return
    
    def cv_plot(self):
         # Plot the cross-validation results
        print()
        print("=== CROSS-VALIDATION PLOT ===")
        print(" Dataset Size:", len(self.data.index))
        print("   Train Size:", len(self.X_train))
        print("    Test Size:", len(self.X_test))
        print("Validate Size:", len(self.X_val))
        print()

        plt.clf()

        cv_params = self.cv_params
        loss_function = self.loss_function
        fig = plt.gcf()
        fig.set_size_inches(5,3)
        plt.errorbar(x=cv_params['iterations'],
                        y=cv_params['test-%s-mean' % loss_function],
                        yerr=cv_params['test-%s-std' % loss_function],
                        fmt='.',
                        ecolor='gray',
                        color='black',
                        capsize=5)
        # plt.title('Cross-validation results')
        plt.xlabel('Iteration')
        plt.ylabel(loss_function )
        plt.tight_layout()
        
        plt.savefig(
            "out/cv-error-%d.pdf" % len(self.X_train), 
            bbox_inches="tight"
        )
        plt.clf()
        return

    def render_tree(self, i_tree = 0):
        # tree = self.model.plot_tree(tree_idx=0, pool=self.dataset_test)
        # tree.render(directory="out")
        import graphviz
        output = "out/savefile.gv"
        print("Number of trees:", self.model.tree_count_)
        self.model.plot_tree(i_tree, self.dataset_train)\
            .save(output)
        with open(output) as f:
            dot_graph = f.read()
            # print(dot_graph)
            dot_graph = dot_graph.replace(", value", " ")
            dot_graph = dot_graph.replace("val =", "")
            dot_graph = dot_graph.replace(" >", "\n>")
            dot_graph = dot_graph.replace("digraph {",
                'digraph {rankdir="LR";')
            dot_graph = dot_graph.replace("e=","e\n=")
            dot_graph = dot_graph.replace("or=","or\n=")
        dot=graphviz.Source(dot_graph)
        dot.render(directory="out")
        
        return

    def force_plot(self, save_type_ext=None):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_val)

        for i in range(5):
            plt.clf()
            shap.waterfall_plot(shap_values[i], show=False)
            fig = plt.gcf()
            fig.set_size_inches(5,3)
            plt.tight_layout() 
            plt.savefig("out/waterfall_%d.pdf" % i)
            # plt.show()

        plt.clf()
        shap.force_plot(shap_values[0],matplotlib =True, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10,3)
        plt.tight_layout() 
        plt.savefig("out/force.pdf")
        return

    def display(self, summary_plot, bar_plot, save_type_ext=None):
        self.summary_plot = summary_plot
        self.bar_plot = bar_plot
        self.save_type_ext = save_type_ext

        # Generate SHAP summary plot (using tree explainer)
        if self.summary_plot:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_val)
            shap.summary_plot(shap_values, self.X_val,
                              show=False, plot_size=[5, 5])
            plt.gcf().axes[-1].set_aspect(100)
            plt.gcf().axes[-1].set_box_aspect(100)
            plt.tight_layout()
            if self.save_type_ext is not None:
                plt.savefig('out/summary_plot' + self.save_type_ext)
                print('Summary plot saved as a ' + self.save_type_ext)
            else:
                plt.show()

        # optional bar chart version (using normal explainer)
        if self.bar_plot:
            bar_explainer = shap.Explainer(self.model)
            bar_shap_values = bar_explainer(self.X_val)
            bar_shap_importance = bar_shap_values.abs.mean(0).values
            sorted_idx = bar_shap_importance.argsort()
            plt.figure(figsize=(5, 2.5))
            plt.barh(range(len(sorted_idx)),
                     bar_shap_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)),
                       np.array(self.X_val.columns)[sorted_idx])
            plt.title('Mean Absolute Impact of SHAP Values')
            plt.tight_layout()  # keeps things from getting cut off
            if self.save_type_ext is not None:
                plt.savefig('out/bar_chart' + self.save_type_ext)
                print('Bar plot saved as ' + self.save_type_ext)
            else:
                plt.show()

    def prediction_error_stats(self) -> dict:
        # Get Stats
        predicted = self.model.predict(self.X_val)
        
        df : pd.DataFrame = self.X_val.copy()
        df["DTC"] = self.y_val
        df["predicted"] = predicted
        df["error"] = np.absolute(predicted - self.y_val)
        df["error_p"] = df["error"] / df["predicted"]
        
        error_p_stats = self.__histogram_stats(
            data = df["error_p"], 
            window_min = 0,
            window_max = 2,
            bin_size = 0.04
        )
    
        error_stats = self.__histogram_stats(
            data = df["error"], 
            window_min = 0,
            window_max = 10,
            bin_size = 0.2
        )
    
        return {
            "error_p" : error_p_stats,
            "error" : error_stats
        }
    

    def __histogram_stats(self, 
        data : pd.Series, 
        window_min : float, 
        window_max : float,
        bin_size : float
    ) -> dict:   
        # Build the dictionary
        hist = {}
        i = window_min
        while i <= window_max:
            hist[i] = 0
            i = np.round(i + bin_size, decimals=2)
            continue

        # Populate histogram
        ls_keys = list(hist.keys())
        for x in data:
            if (x < window_min) or (x > window_max):
                continue

            for i in range(len(ls_keys)):
                bin = ls_keys[i]
                if x == bin:
                    hist[bin] += 1
                    break

                next_bin = ls_keys[i+1]
                if (x >= bin) and (x < next_bin):
                    hist[bin] += 1
                    break
                continue
            continue
            
        return hist

class Point:
    def __init__(self, x : float, y : float):
        self.x = x
        self.y = y

class Test():

    def __init__(self):
        self.params_csv = "data/old_5d/params_5d.csv"
        self.scores_csv = "data/old_5d/scores_5d.csv"
        self.partial_dataset(frac = 1.0, plots=True)
        # self.generate_validation_data(out_fn = "validation_test/v2.json")
        # self.validation_query()
        # self.collect_error_stats()
        # self.interpret_error_stats()
        return

    def interpret_error_stats(self,
        total_tests : int,
        json_fn : str = "validation_tests/error_p.json",
        fig_error_hist_fn : str = "validation_tests/error-hist.pdf",
        fig_error_sum_fn : str = "validation_tests/error-sum-p_error.pdf"
    ):
        #  Read from file
        with open(json_fn,"r") as f:
                data = json.load(f)
        
        # Prepare Data
        histograms = {}
        for frac, tests in data.items():
            # Get average of all tests
            n = len(tests.keys())

            # Prepare Dict
            test_avg = {}
            for key in tests['0'].keys():
                test_avg[float(key)] = 0
            
            # Get sums
            for hist in tests.values():
                for bin, count in hist.items():
                    test_avg[float(bin)] += count
            
            # Take Average
            for bin in test_avg.keys():
                test_avg[bin] /= n
            
            # Add to the dataset
            histograms[float(frac)] = test_avg
            continue

        # Get proportion stats to make the histograms comparable
        for frac, hist in histograms.items():
            frac = float(frac)
            n_validation_samples = int(frac * 99604 * .2)
            for bin, n in hist.items():
                hist[bin] = n/n_validation_samples
            continue

        #  Graph Histogram
        fig = plt.figure()
        ax = fig.add_subplot()

        # How many tests are in each fraction
        total_tests = 99604
        legend_labels = ["p=%3.2f, n=%4d" % (frac, float(frac) * total_tests) \
                         for frac in histograms.keys()]

        cmap = matplotlib.colormaps["binary"]

        # Generate Histograms
        for frac, hist in histograms.items():
            points = self.histogram_bars(
                bins = list(hist.keys()), 
                proportion = list(hist.values()),
                bar_width = 0.2
            )
            x = points.T[0]
            y = points.T[1]
            rgba = cmap(float(frac))
            ax.plot(x, y, color=rgba)
            continue

        ax.set_xlabel("error")
        ax.set_ylabel("proportion")
        ax.set_ylim(bottom=0)
        ax.legend(legend_labels,loc="upper right")
        fig = plt.gcf()
        fig.set_size_inches(5,2.5)
        plt.tight_layout()
        plt.savefig(fig_error_hist_fn)


        # Error under curve
        ls_p_error = []
        for frac, hist in histograms.items():
            p_error = {}
            for p_max in [1., 2.5, 5]:
                p_error[p_max] = 0
                for bin, prop in hist.items():
                    if bin <= p_max:
                        p_error[p_max] += prop
                    continue
                continue
            ls_p_error.append( pd.Series(p_error, name=frac) )
            continue
        df = pd.DataFrame(ls_p_error)
        df.index.name = "% Results"
        print(df)


        # Plot line graph
        print()

        legend_labels = ["sum(p_error <= %.2f)" % p_error_max \
                         for p_error_max in df.columns]

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        for p_error_max in df.columns:
            rgba = cmap(p_error_max*2)
            ax.plot(df.index, df[p_error_max], color=rgba)
            continue
        ax.set_xlabel("Proportion of total tests.")
        ax.set_ylabel("sum(p_error)")
        ax.set_xticks(df.index)
        ax.legend(legend_labels,loc="best", prop={'size':8})
        fig = plt.gcf()
        fig.set_size_inches(5,2.5)
        plt.tight_layout()
        plt.savefig(fig_error_sum_fn)
        return


    def interpret_error_p_stats(self,
        total_tests : int,
        json_fn : str = "validation_tests/error_p.json",
        fig_error_hist_fn : str = "validation_tests/error-hist.pdf",
        fig_error_sum_fn : str = "validation_tests/error-sum-p_error.pdf"
    ):
        #  Read from file
        with open(json_fn,"r") as f:
                data = json.load(f)
        
        # Prepare Data
        histograms = {}
        for frac, tests in data.items():
            # Get average of all tests
            n = len(tests.keys())

            # Prepare Dict
            test_avg = {}
            for key in tests['0'].keys():
                test_avg[float(key)] = 0
            
            # Get sums
            for hist in tests.values():
                for bin, count in hist.items():
                    test_avg[float(bin)] += count
            
            # Take Average
            for bin in test_avg.keys():
                test_avg[bin] /= n
            
            # Add to the dataset
            histograms[float(frac)] = test_avg
            continue

        # Get proportion stats to make the histograms comparable
        for frac, hist in histograms.items():
            frac = float(frac)
            n_validation_samples = int(frac * 99604 * .2)
            for bin, n in hist.items():
                hist[bin] = n/n_validation_samples
            continue

        #  Graph Histogram
        fig = plt.figure()
        ax = fig.add_subplot()

        # How many tests are in each fraction
        legend_labels = ["p=%3.2f, n=%4d" % (frac, float(frac) * total_tests) \
                         for frac in histograms.keys()]

        cmap = matplotlib.colormaps["binary"]

        # Generate Histograms
        for frac, hist in histograms.items():
            points = self.histogram_bars(
                bins = list(hist.keys()), 
                proportion = list(hist.values()),
                bar_width = 0.04
            )
            x = points.T[0]
            y = points.T[1]
            rgba = cmap(float(frac))
            ax.plot(x, y, color=rgba)
            continue

        ax.set_xlabel("error")
        ax.set_ylabel("proportion")
        ax.set_ylim(bottom=0)
        ax.legend(legend_labels,loc="upper right")
        fig = plt.gcf()
        fig.set_size_inches(5,2.5)
        plt.tight_layout()
        plt.savefig(fig_error_hist_fn)


        # Error under curve
        ls_p_error = []
        for frac, hist in histograms.items():
            p_error = {}
            for p_max in [0.1, 0.25, 0.5]:
                p_error[p_max] = 0
                for bin, prop in hist.items():
                    if bin <= p_max:
                        p_error[p_max] += prop
                    continue
                continue
            ls_p_error.append( pd.Series(p_error, name=frac) )
            continue
        df = pd.DataFrame(ls_p_error)
        df.index.name = "% Results"
        print(df)


        # Plot line graph
        print()

        legend_labels = ["sum(p_error <= %.2f)" % p_error_max \
                         for p_error_max in df.columns]

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        for p_error_max in df.columns:
            rgba = cmap(p_error_max*2)
            ax.plot(df.index, df[p_error_max], color=rgba)
            continue
        ax.set_xlabel("Proportion of total tests.")
        ax.set_ylabel("sum(p_error)")
        ax.set_xticks(df.index)
        ax.legend(legend_labels,loc="best", prop={'size':8})
        fig = plt.gcf()
        fig.set_size_inches(5,2.5)
        plt.tight_layout()
        plt.savefig(fig_error_sum_fn)
        return

    
    def histogram_bars(self, 
            bins : list[float], 
            proportion : list[int], 
            bar_width : float
        ) -> np.ndarray:
        # Find the values between bins
        points = [Point(bins[0] - bar_width/2, 0)]
        for i in range(len(bins)):
            bin = bins[i]
            prop = proportion[i]

            p0 = Point(bin - bar_width/2, prop)
            # p1 = Point(bin, prop)
            p2 = Point(bin + bar_width/2, prop)

            points.append(p0)
            points.append(p2)
            continue
        points.append(Point(bins[-1] + bar_width/2, 0))

        return np.array([[p.x, p.y] for p in points])

    def collect_error_stats(self, 
            error_fn : str = "validation_tests/error.json", 
            error_p_fn : str = "validation_tests/error_p.json"
        ):
        all_stats = {}
        all_stats_p = {}
        for frac in [.1,.25,.5,.75,1.]:
            hist_stats = {}
            hist_stats_p = {}
            for i in range(30):
                rm = self.partial_dataset(frac = frac, exclude_0=True)
                es = rm.prediction_error_stats()
                hist_stats[i] = es["error"]
                hist_stats_p[i] = es["error_p"]
                continue
            all_stats[frac] = hist_stats
            all_stats_p[frac] = hist_stats_p
            continue

        with open(error_fn, 'w') as f:
            json.dump(all_stats, f, indent=2)
        with open(error_p_fn, 'w') as f:
            json.dump(all_stats_p, f, indent=2)
        return


    def validation_query(self, 
            total_tests : int,
            json_fn : str = "validation_tests/data.json",
            fig_fn : str = "validation_tests/validation-comparison.pdf"
        ):
        
        #  Read Data
        with open(json_fn, "r") as f:
            data = json.load(f)
    
        #  Consolidate test data
        for frac in data.keys():
            # Get mean of RMSE
            all_dfs = []
            for n, test in data[frac].items():
                df = pd.DataFrame({
                    "n" : n,
                    "iteration" : test["x"],
                    "RMSE" : test["y"]
                })
                all_dfs.append(df)
            df = pd.concat(all_dfs).reset_index(drop=True)
            
            # Get mean for each iteration
            rmse_mean = [df[df["iteration"] == i]["RMSE"].mean() \
                for i in df["iteration"].unique()]
            
            df = pd.DataFrame({
                "iteration" : [_ for _ in range(100)],
                "RMSE" : rmse_mean,
            })
            data[frac] = df
            continue

        # How many tests are in each fraction
        legend_labels = ["p=%s, n=%4d" % (frac, float(frac) * total_tests) \
                         for frac in data.keys()]

        # generate graphs
        fig = plt.figure(figsize=(5,3.2))
        ax = fig.add_subplot()

        cmap = matplotlib.colormaps["binary"]

        for frac, df in data.items():
            
            x = df["iteration"]
            y = df["RMSE"]
            rgba = cmap(float(frac))
            ax.plot(x,y, color=rgba)
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE")
        ax.legend(legend_labels,loc="upper right")
        plt.tight_layout()

        plt.savefig(fig_fn)

        return

    def generate_validation_data(self, 
            out_fn : str = "validation_tests/data.json"):
        data = {}
        for i in range(10):
            frac = np.round(i*.1+.1, decimals=1)
            data_n = {}
            for n in range(30):
                rm = self.partial_dataset(frac)
                data_n[n] = {
                    "x" : rm.cv_params["iterations"].to_list(),
                    "y" : rm.cv_params["test-%s-mean" % rm.loss_function]\
                        .tolist()
                }
                continue
            data[frac] = data_n
            continue

        with open(out_fn, "w") as f:
            json.dump(data, f, indent=4)
        return



    def partial_dataset(self,
            frac : float, 
            exclude_0 : bool = False,
            plots : bool = False
        ) -> RegressionModel:
        print("Version:", tf.__version__)
        gpu_devices = tf.config.list_physical_devices("GPU")
        print(gpu_devices)

        # Read in data
        params = ["dut_speed","dut_decel","dut_dist","foe_speed",
                    "foe_dist","envelope_id"]
        params_df = pd.read_csv(self.params_csv)
        print("n params:", len(params_df.index))

        scores = ["dtc"]
        scores_df = pd.read_csv(self.scores_csv)
        scores_df = scores_df[scores]
        print("n scores:", len(scores_df.index))

        df = pd.merge(params_df, scores_df, left_index=True, right_index=True)
        # df = df[df["envelope_id"]  < 100]

        # Exclude DTC = 0
        if (exclude_0):
            df = df[df["dtc"] != 0]

        # Find categorical data
        df["envelope_id"] = df["envelope_id"].astype(str)
        cat_feat_indices = [df.columns.tolist().index("envelope_id")]
        print("Categorical Features:", df.columns[cat_feat_indices].tolist())

        # Scramble data
        df = df.sample(frac=frac)

        # Grab the score
        score_s = df["dtc"]

        # Drop the score field from the params df
        params_df = df.drop(columns = [score_s.name], axis=1)
       
        # Build the regression model
        rm = RegressionModel(params_df, score_s.values.tolist(),
                             cat_feat_indices)
        
        if plots:
            rm.cv_plot()
            rm.render_tree(i_tree=99)
            rm.display(True, True, save_type_ext=".pdf")
            rm.force_plot()
        return rm


if __name__ == "__main__":
    Test()
