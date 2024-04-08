import itertools as itt
from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns


def plot_pairs_samples(
    X: pd.DataFrame,
    metadata_table: pd.DataFrame,
    map_colors: Dict[str, str],
    sample_unformat: str = "{group}{replicate}",
    main_title: str = "",
    savefigfile: Union[str, None] = None,
    **kwargs,
):
    """Plot correlation of pairs of samples established from the <metadata_table>"""

    pairs = (
        metadata_table.groupby("group")["replicate"]
        .apply(lambda v: pd.Series(list(itt.combinations(v, 2))))
        .reset_index()
    )

    Tot = pairs.shape[0]
    Cols = kwargs.get("Cols", 3)
    Rows = Tot // Cols
    Rows += Tot % Cols
    Position = range(1, Tot + 1)

    fig = plt.figure(figsize=(20, 12))

    for i, (pair_label, row) in enumerate(pairs.iterrows()):
        group_name = str(row["group"])
        color_group = map_colors[group_name]
        rep1 = row["replicate"][0]
        rep2 = row["replicate"][1]

        pos = Position[i]
        ax = fig.add_subplot(Rows, Cols, pos)
        sns.scatterplot(
            data=X,
            x=sample_unformat.format(group=group_name, replicate=rep1),
            y=sample_unformat.format(group=group_name, replicate=rep2),
            color=color_group,
            ax=ax,
        )
        # ax.set_aspect("equal")

        ax.set_xlabel(f"Rep {rep1}")
        ax.set_ylabel(f"Rep {rep2}")

        r, p = scipy.stats.spearmanr(
            X[sample_unformat.format(group=group_name, replicate=rep1)],
            X[sample_unformat.format(group=group_name, replicate=rep2)],
        )
        ax.set_title(f"{group_name}\n(r={r:.3}, pval={p:.3})", y=1.1)

    plt.suptitle(main_title, y=1.05)
    plt.tight_layout()

    if savefigfile:
        plt.savefig(savefigfile)

    plt.show()
