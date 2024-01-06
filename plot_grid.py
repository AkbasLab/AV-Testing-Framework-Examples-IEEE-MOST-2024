import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import pandas as pd
import numpy as np

class GridPlot:
    def __init__(self):
        params_df = pd.read_csv("data/old_5d/params_5d.csv")
        scores_df = pd.read_csv("data/old_5d/scores_5d.csv")\
            .drop(columns="envelope_id")
        
        df = pd.concat([params_df, scores_df], axis=1)
        df = df[df["dtc"] >= 0] 
        # df = df.iloc[:10_000]

        # 
        features = ["dut_speed", "dut_decel", "dut_dist",
                        "foe_speed", "foe_dist"]
        score_feats = ["envelope_id", "dtc"]

        for score_feat in score_feats:
            self.lattice_plot(df, features, score_feat, 
                              fig_size=(4,4), dpi=250)
            # break
        return

        
        
    def lattice_plot(self,
            df : pd.DataFrame,
            features : list[str],
            score_feat : str,
            fig_size : list[float, float],
            dpi : int             
        ):
        plt.clf()

        # Colors
        cmap_id = "gist_heat"
        cmap = mpl.colormaps[cmap_id]
        
        # Normalize score
        scores_norm = (df[score_feat]-df[score_feat].min())\
                        /(df[score_feat].max()-df[score_feat].min())
        df["color"] = scores_norm.apply(cmap)
        a = df[score_feat].min()
        b = df[score_feat].max()
        norm_ticks = [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
        score_ticks = [np.round((b-a)*x + a, decimals=2) for x in norm_ticks]
        
        # Make the plots
        plt.rc("font", size=8)
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        nrow = len(features)
        ncol = len(features)
        for irow, y_feat in enumerate(features):
            for icol, x_feat in enumerate(features):
                ax : Axes = plt.subplot2grid((ncol, nrow), (irow, icol))

                # Setup labels
                is_left = icol == 0
                is_bottom = irow == nrow - 1

                ax.tick_params(
                    left = is_left, 
                    right = False , 
                    labelleft = is_left , 
                    labelbottom = is_bottom, 
                    bottom = is_bottom
                )

                if is_left:
                    ax.set_ylabel(y_feat)
                if is_bottom:
                    ax.set_xlabel(x_feat)

                # Skip same plot
                if x_feat == y_feat:
                    ax.set_facecolor("grey")
                    continue

                # Make scatter plot
                ax.scatter(
                    df[x_feat],
                    df[y_feat],
                    color = df["color"],
                    marker=",",
                    s=(72/dpi)**2
                )
                continue
            continue

        
        # plt.tight_layout()
        plt.subplots_adjust(hspace=0,wspace=0)
        

        # Put the colorbar on the grid
        cbar_ax = fig.add_axes([0.91, 0.12, 0.01, .75])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(0, 1), 
                cmap=cmap_id
            ),
            cax=cbar_ax, 
            orientation="vertical",
            label=score_feat,
        )
        cb.set_ticks(norm_ticks)
        cb.set_ticklabels(score_ticks)

        

        plt.savefig("eda/%s.png" % score_feat, bbox_inches="tight")
        return



    
if __name__ == "__main__":
    GridPlot()