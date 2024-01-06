import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.textpath

class Envelope_EDA:
    def __init__(self):
        
        self.hhh()
        
        return
    
    def bbb(self):
        df_5d = self.count_collisions("5d")
        df_mc = self.count_collisions("mc")
        
        for df in [df_5d, df_mc]:
            plt.plot(
                df.index,
                df["collisions_so_far"]
            )

        plt.savefig("eda/hhh.png")

    def count_collisions(self, dataset :str) -> pd.DataFrame:
        if dataset == "5d":
            params_df = pd.read_csv("data/5d/params_5d.csv")
            scores_df = pd.read_csv("data/5d/scores_5d.csv")\
                .drop(columns="envelope_id")
        elif dataset == "mc":
            params_df = pd.read_csv("data/5d_mc/params_mc.csv")
            scores_df = pd.read_csv("data/5d_mc/scores_mc.csv")
        df = pd.concat([params_df, scores_df], axis=1)
        
        # Count collision
        n_collisions = 0
        ls_n_collisions = []
        for is_collision in df["collision"]:
            n_collisions += is_collision
            ls_n_collisions.append(n_collisions)
        
        df["collisions_so_far"] = ls_n_collisions
        return df

    
    def hhh(self):
        params_df = pd.read_csv("data/old_5d/params_5d.csv")
        scores_df = pd.read_csv("data/old_5d/scores_5d.csv")\
            .drop(columns="envelope_id")
        
        df = pd.concat([params_df, scores_df], axis=1)
        print(df)

        n_irrelevent = len(df[df["dtc"] < 0].index)
        n_collision = df["collision"].sum()
        n_dtc = len(df[df["dtc"] > 0].index)
        total = len(df.index)
        print("dtc == -1: %5d %.4f" % (n_irrelevent, n_irrelevent/total))
        print("dtc ==  0: %5d %.4f" % (n_collision, n_collision/total))
        print("dtc  >  0: %5d %.4f" %  (n_dtc, n_dtc/total))
        
        df.hist(column = "dtc", grid=False, color="black", bins=30)
        ax = plt.gca()
        ax.set_title(None)
        fig = plt.gcf()
        fig.set_size_inches(5,2.5)

        plt.savefig("eda/dtc-hist.pdf", bbox_inches="tight")
        return
    
if __name__ == "__main__":
    Envelope_EDA()