#! /usr/bin/env python
# coding:utf-8

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt



def create_regular_grid_axes(N_tot, N_cols, height_row, width):
    """ Create a figure with a grid of N_tot axes with N_cols.
    
    Given N_tot a number of axes to generate, and N_cols the maximum
    number of axes on a row, this function generates the list of axes
    with the correct layout.
    
    To define the size of the figure, height_row and width are used.
    
    In:
        N_tot (int):
        N_cols (int):
        height_row (int):
        width (int):
        
    Return:
        (fig, axs)
        
    """
    N_rows = N_tot // N_cols
    N_rows += N_tot % N_cols
    Position = range(1, N_tot+1)
    
    fig = plt.figure(figsize = (width, height_row * N_rows))
    axs = []
    for pos in Position:
        axs.append(fig.add_subplot(N_rows, N_cols, pos))
        
    return (fig, axs)




class Multilayer_DE_results():
    """
    """
    colors_DEstatus = {"not_DE":"#BBBBBB",
                       "UP_{layer1}":"#ff00e5", # PURPLE
                       "DOWN_{layer1}":"#ff00e5", # PURPLE
                       "UP_{layer2}":"#ffbb00", # YELLOW
                       "DOWN_{layer2}":"#ffbb00", # YELLOW
                       "DE_discordant":"#1f26ad", # Deep purple
                       "UP_BOTH":"#ff0000", # RED
                       "DOWN_BOTH": "#ff0000", # RED
                       }

    def __init__(self):
        self.layers = {}

        self.name_merged = None
        self.merged_table = None
        self.merged_table_info = None

        self.filters = {}



    def __repr__(self):
        repr_str = ""
        return repr_str

    def add_layer(self, name, DE_results_object):
        """ Add the object <DE_results_object> into stored layers under <name>.
        """
        if name in self.layers:
            raise ValueError(f"{name} already found in layers.")

        if not isinstance(DE_results_object, DE_results):
            raise ValueError("Provided DE_results_object is not of class <DE_results>.")

        self.layers[name] = DE_results_object

    def get_layer(self, name):
        return self.layers[name]


    def merge_layers(self, name1, name2, name_merged, mapping_dict=None):
        """ Merge two stored layers on their index.

        The mapping table should map indices from <name1> to indices of <name2> in a 1-1 way.


        """
        if not name1 in self.layers:
            raise ValueError(f"{name1} not found in layers")

        if not name2 in self.layers:
            raise ValueError(f"{name1} not found in layers")

        if (self.layers[name1].table.index.isin(self.layers[name2].table.index.values)).sum()==0 and mapping_dict is None:
            raise ValueError(f"None of indices from {name1} were found in {name2}.")


        # First part: merging

        if mapping_dict:
            merged_table = self.layers[name1].table.copy().add_prefix(name1+".")
            merged_table[name2+"_mapped"] = list(map(lambda v: mapping_dict.get(v, None), merged_table.index.values))

            if merged_table[name2+"_mapped"].isnull().all():
                raise ValueError(f"None of indices from {name1} were found in the mapping dict.")

            merged_table = merged_table.loc[~merged_table[name2+"_mapped"].isnull(),:].copy()

            dropped_1 = self.layers[name1].table.loc[
                            ~(self.layers[name1].table.index.isin(merged_table.index.values)),
                            :].index.values

            merged_table = pd.merge(merged_table.loc[~merged_table.index.isin(dropped_1),:],
                                    self.layers[name2].table.copy().add_prefix(name2+"."),
                                    left_on=name2+"_mapped",
                                    right_index=True,
                                    how="inner"
                                   )
                                   
            dropped_2 = self.layers[name2].table.loc[
                            ~(self.layers[name2].table.index.isin(merged_table[name2+"_mapped"].values)),
                            :].index.values

            #return merged_table

        else:
            merged_table = pd.merge(self.layers[name1].table.add_prefix(name1+"."),
                                    self.layers[name2].table.add_prefix(name2+"."),
                                    left_index=True,
                                    right_index=True,
                                    how="inner",
                                    validate="one_to_one",
                                   )

            dropped_1 = self.layers[name1].table.loc[
                            ~(self.layers[name1].table.index.isin(merged_table.index.values)),
                            :].index.values
            dropped_2 = self.layers[name2].table.loc[
                            ~(self.layers[name2].table.index.isin(merged_table.index.values)),
                            :].index.values



        # Second part : annotate the genes from the lFC and p.adj columns kept from each layer.

        merged_table["merged.concordance"] = merged_table.apply(
                                        lambda row: self._label_direction_expression(
                                                        row[name1+".logFC"],
                                                        row[name2+".logFC"]), axis=1)

        #TODO : here I might want to use a different "significant" column than the original
        # one in the DE tables... So this may need to be an input of the table.
        merged_table["merged.DE_status"] = merged_table.apply(
                                            lambda row: self._label_merged_DE_status(
                                                            row, name1, name2), axis=1)


        self.merged_table = merged_table
        self.name_merged = name_merged
        self.merged_table_info = {"merged_layers":(name1,name2),
                                  "info":{f"droppedFrom_{name1}":dropped_1,
                                          f"droppedFrom_{name2}":dropped_2
                                         }
                                 }


    def get_merged_layer(self, full=True):
        if full:
            table_filters = self._generate_filters_table()
            if table_filters is None:
                return self.merged_table

            return pd.concat([self.merged_table,
                              table_filters],
                             axis = 1)
        else:
            return self.merged_table

    def info_merged_layer(self):
        """ Return a formatted string of information about the merged-layers <name_merged>.
        """
        name1, name2 = self.merged_table_info["merged_layers"]
        N_merged = self.merged_table.shape[0]
        N_missing_1 = len(self.merged_table_info["info"][f"droppedFrom_{name1}"])
        N_missing_2 = len(self.merged_table_info["info"][f"droppedFrom_{name2}"])

        info_str = f"Merged layers : {name1} and {name2}\n"
        info_str += (f"N genes : {N_merged} ; {N_missing_1} missing from {name1}, "
                     f"{N_missing_2} missing from {name2}\n")

        return info_str

    def get_matrix_agreement_DE_status(self):
        """ Using "DE_status" columns from each layer, count the number of genes per pair of status.

        Pairs of status may be in agreement (e.g. : both layers identify a gene as "UP", increasing the (UP,UP) group of 1),
        or in disagreement (e.g. : a gene is "UP" in one and "notSign" in the second, increasing the (UP,notSign) group of 1).

        Args:
            self
            name_merged (str) : name of the merged layer to investigate.

        Returns:
            pandas.DataFrame:
        """
        index_labels = ["DOWN","UP","notSign"]
        name1, name2 = self.merged_table_info["merged_layers"]
        tmp = self.merged_table.filter(regex="DE_status"
                                      ).pivot_table(index=f"{name1}.DE_status",
                                                    columns=f"{name2}.DE_status",
                                                    aggfunc="size"
                                                    ).reindex(index=index_labels,
                                                              columns=index_labels
                                                            )
        tmp = tmp.replace(np.nan, 0).astype(int)
        return tmp



    def _generate_filters_table(self):
        """ Generate a table of bool values with each column representing an added filter.
        """
        if len(self.filters)==0:
            print("No filters found.")
            return None

        return pd.DataFrame(self.filters, index=self.merged_table.index.values)


    def get_dict_filters(self):
        """ Return a dictionary of all the added filters with associated genes lists.

        If you want to re-create a DE object from a subset of the table, this function allows you
        to directly generate a dictionary of all the available filters which were applied in the first table.

        This will map {"name_filter":[list_genes]} ; where list genes will contain all the genes from the
        original table that matched the initial filter.

        You can then easily loop through the dictionary to use <add_filtering_geneset> on the new DE_results.
        """
        tmp = self._generate_filters_table()

        filters_dict = {}
        for filter_name, bool_genes_dict in tmp.to_dict().items():
            filters_dict[filter_name] = [k for k, v in bool_genes_dict.items() if v]

        return filters_dict



    def add_filtering_geneset(self, name_filter, list_genes):
        """ From a list of genes, store a bool vector identifying genes from the DE table.

        The "list_genes" should contain gene names which will be assigned a "True" in the
        DE table ; genes from the DE table not found in the list will be assigned a "False".

        You can reverse this True/False when applying the filter using the <apply_filter> method.
        """

        res = self.merged_table.index.isin(list_genes)
        self.filters[name_filter] = res



    def apply_filter(self, name_filter, reverse=False):
        """ Apply one of the filters previously added.
        """
        if name_filter not in self.filters.keys():
            print(f"Requested filter {name_filter} not found.")
            return None

        if name_filter not in self.filters:
            print("Filter not recognized.")
            return None

        if reverse:
            return self.merged_table.loc[~self.filters[name_filter],]
        else:
            return self.merged_table.loc[self.filters[name_filter],]



    def compose_filters(self, name_filter, filter_operations):
        """ Create a composite filter basing on stored filters.

        "filter_operations" should be a string, containin names of pre-existing filters,
        and composed as a query for a pandas.DataFrame table, ie:
        - "&" will get the intersection of two boolean vectors
        - "|" will get the union
        - parenthesis are accepted
        - inverting a boolean is possible through the character "~"

        Any missing filter will raise an error from Pandas, handled to return None ; in that case no filter is created.

        """
        try:
            tmp = self._generate_filters_table()
            res = tmp.query(filter_operations)
            self.filters[name_filter] = self.merged_table.index.isin(res.index.values)

        except pd.core.computation.ops.UndefinedVariableError as e:
            print(f"Filter not recognized when parsing the composite filter: {e}")
            print("No filter created.")
            return



    def _label_direction_expression(self, logFC1, logFC2):
        """ Indicate concordance between the layers from the logFC values.

        The sign of the two logFC is used to define whether:
        - the layers are *discordant* (oposite signs)
        - the layers are concordant *positive*
        - the layers are concordant *negative*

        """
        if np.sign(logFC1) * np.sign(logFC2) < 0:
            return "discordant"
        else:
            if np.sign(logFC1)>0:
                return "positive"
            else:
                return "negative"


    def _label_merged_DE_status(self, row, name1, name2):
        """

        This function defines from the <name1>.DE_status and <name2>.DE_status a single label as follow:
        - (UP,UP) : merged.UP_BOTH
        - (DOWN,DOWN) : merged.DOWN_BOTH
        - (UP,notSign|belowLFCThresh) : merged.UP_name1
        - (DOWN,notSign|belowLFCThresh) : merged.DOWN_name1
        - same for name2 with inverted labels
        - (notSign|belowLFCThresh,notSign|belowLFCThresh) : merged.not_DE
        - (UP,DOWN) or (DOWN,UP) : merged.DE_discordant

        """

        DE_status1, DE_status2 = row[name1+".DE_status"], row[name2+".DE_status"]

        if DE_status1 == DE_status2:
            if DE_status1 == "UP":
                return "merged.UP_BOTH"

            elif DE_status1 == "DOWN":
                return "merged.DOWN_BOTH"
            else:
                # No difference between notSign and belowLFCThresh
                return "merged.not_DE"

        else:
            if ((DE_status1 == "UP" and DE_status2 == "DOWN") or (DE_status1 == "DOWN" and DE_status2 == "UP")):
                return "merged.DE_discordant"

            elif ((DE_status1 in ("UP", "DOWN")) and (DE_status2 in ("notSign","belowLFCThresh"))):
                return f"merged.{DE_status1}_{name1}"

            elif ((DE_status1 in ("notSign","belowLFCThresh")) and (DE_status2 in ("UP","DOWN"))):
                return f"merged.{DE_status2}_{name2}"

            else:
                # No difference between notSign and belowLFCThresh
                return "merged.not_DE"


    def scatterplot_LFC_merged(self, x, y,
                               label_mapping_dict=None, highlight=None,
                               ax=None, show_plot=True, savefig_file=None):
        """
        """
        if not "merged.DE_status" in self.merged_table.columns:
            raise ValueError("You need to call the method <label_DE_status> before running this method.")

        tmp = self.merged_table.copy()
        tmp["status"] = tmp["merged.DE_status"].str.replace("merged.","")

        if ax is not None:
            ax1 = ax
        else:
            fig = plt.figure(figsize=(16,8))
            ax1 = fig.add_subplot(1,1,1)

        labeled_colors_DEstatus = {k.format(**{"layer1":x,"layer2":y}):v for k,v in self.colors_DEstatus.items()}

        sns.scatterplot(data=tmp,
                        x=f"{x}.logFC",
                        y=f"{y}.logFC",
                        alpha=0.6,
                        hue="status",
                        hue_order=list(labeled_colors_DEstatus.keys())[::-1],
                        palette=labeled_colors_DEstatus,
                        ax=ax1)

        if highlight is not None and highlight in self.filters:
            print(f"Applying filter: {highlight}")
            for gene_id, row in self.apply_filter(highlight).iterrows():
                gene_name = label_mapping_dict.get(gene_id,'') if label_mapping_dict else gene_id
                ax1.text(row[f"{x}.logFC"],#*1.05,
                         row[f"{y}.logFC"],#*1.05,
                         gene_name,
                         horizontalalignment='left',
                         size=14, color='black', weight='normal')

        ax1.legend(bbox_to_anchor=(1,1))

        #ax1.set_title("DE genes without genes DE in emptyVect")
        plt.tight_layout()

        if savefig_file:
            plt.savefig(savefig_file)

        if show_plot:
            plt.show()
        else:
            return ax1







class DEA_experiment_results():
    """ Handle read counts and differential expression results.
    """
    def __init__(self, RC_dict, samples_colors):
        self.read_counts = {}
        for k, v in RC_dict.items():
            self.read_counts[k] = v
            
        self.samples_colors = samples_colors
        
        
    def _pairplot_RC(self, sample1, sample2, rc_type, ax=None):
        """ 
        """
        if not rc_type in self.read_counts:
            return None
        
        if not ax:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(1,1,1)
            
        
    def plot_pairs_samples(self, main_title="", savefigfile=None):
        """ From a long-format data-frame X, do a scatterplot basing on groups from the "map_colors" dict.

        map_colors should contain names of all conditions of interest for which measurements are available.

        From these names, the replicates (which should form a pair) are retrieved and used for x and y axes.

        A list of plots is thus created from all names.

        Along the scatter plot, the Spearman correlation is calculated.
        """
        fig, axs = ML_visualization.create_regular_grid_axes(
                        N_tot=len(map_colors), 
                        N_cols=4, 
                        height_row=4,
                        width=20)

        for i, (group_name, color_group) in enumerate(map_colors.items()):
            ax = axs[i]
            sns.scatterplot(
                data=X,
                x=group_name+"_a",
                y=group_name+"_b",
                color=color_group,
                ax=ax
            )
            ax.set_aspect("equal")
            ax.set_xlabel("Rep a")
            ax.set_ylabel("Rep b")

            r, p = scipy.stats.spearmanr(X[group_name+"_a"],
                                        X[group_name+"_b"]
                                       )
            ax.set_title(group_name+"\n(r={:.3}, pval={:.3})".format(r, p), y=1.1)



        plt.suptitle(main_title, y=1.05)

        if savefigfile is not None:
            plt.savefig(savefigfile)
        plt.show()



class DE_results():
    """ Handler of pairwise comparison DEA results

    Caution :
    * condition1 and condition2 should follow the order which was used during the DEA.
    * thresh_LFC and thresh_pval should correspond to the thresholds used during the DEA.
      Additional filters on these values should be defined through the appropriate methods.

    """

    expected_columns = ["logFC","pval","padj"]

    ordered_DE_status = ("notSign", "belowThreshLFC", "UP", "DOWN")


    color_diffexp = ("#BBBBBB","#ce3131")


    def __init__(self, table, condition1, condition2,
                 thresh_LFC,
                 thresh_pval,
                 path,
                 process_significant="check",
                ):
        """ Initialization

        Args:
            table (pandas.DataFrame) : a table of counts, with unique gene IDs as index, and loFC, pval, padj
                                       as mandatory columns. "significant" may also be provided, and will be checked
                                       against the internal "significant" computation method.
            condition1 (str) : name of condition1 from the pairwise comparison
            condition2 (str) : name of condition2 from the pairwise comparison
            thresh_LFC (numeric) : value used for the logFoldChange threshold, during the DEA.
            thresh_pval (numeric) : value used for the adjustment of p-values during the DEA.
            process_significant ()

        """
        options_process_significant = ["check", "compute", "no"]
        if not process_significant in options_process_significant:
            raise ValueError(f"<process_significant> should be among {options_process_significant}.")

        if any([c not in table.columns.values for c in self.expected_columns]):
            raise ValueError(f"One of the expected columns was not found : {self.expected_columns}")

        if table["padj"].isnull().any():
            raise ValueError("A missing value was found in the padj column")

        if table["logFC"].isnull().any():
            raise ValueError("A missing value was found in the logFC column")

        additional_columns = []

        for c in table.columns:
            if c not in self.expected_columns:
                additional_columns.append(c)

        self.table = table.loc[:,self.expected_columns+additional_columns].copy()

        self.condition1 = condition1
        self.condition2 = condition2
        self.name = f"{self.condition1} vs {self.condition2}"
        self.thresh_LFC = thresh_LFC
        self.thresh_pval = thresh_pval
        self.path = path

        # Now add "significant" labels to the genes in the table.
        if "significant" in self.table.columns.values:
            if process_significant=="check":
                matching_significant = (self.table["significant"] ==
                                        self._annotate_table_significant(in_place=False))

                if not (matching_significant).all():
                    raise ValueError(("The provided \"significant\" column does not match the internal "
                                      f"{(~matching_significant).sum()} differing elements."
                                     ))

            elif process_significant == "compute":
                print("Warning: <process_significant> was set to 'compute' while column was found ; replacing values.")
                self._annotate_table_significant()

        else:
            if process_significant in ["no","check"]:
                print((f"Warning: <process_significant> was set to {process_significant} "
                        "while column was not found ; generating values."))

            self._annotate_table_significant()


        # And add the "gene DE status"
        self._annotate_table_DE_status()


        # Finally, create a dictionary that will store filters.
        # These filters should be lists of genes which can be used
        # to filter genes of interests.
        self.gene_filters = {"significant":self.table["significant"].values
                            }

        self.colors_filters = {"significant":self.color_diffexp}

        # Since the "significant" vector is stored in the dict, it can be dropped from the table.
        self.table = self.table.drop("significant", axis=1)

        # Last check : number of DOWN and UP should match the number of significant genes:
        if not (sorted(self.table.loc[self.table["DE_status"].isin(["UP","DOWN"]),:].index.values) ==
                sorted(self.get_significant().index.values)):
            raise ValueError("Significant genes list is not the same as UP/DOWN list.")




    def __repr__(self):
        repr_str = ""
        repr_str += (f"DE experiment : {self.condition1} vs {self.condition2} "
                     f"with thresholds pVal≤{self.thresh_pval} and |LFC|≥{self.thresh_LFC}"
                     "\n\n"
                    )

        repr_str += f"Number of genes: {self.table.shape[0]:,} "
        repr_str += f"of which {self.apply_filter('significant').shape[0]:,} are DE\n"

        DE_status_str = "\n".join([f"* {k}: {v:,}"
                                   for k, v in self.table["DE_status"].value_counts(
                                   ).reindex(self.ordered_DE_status).replace(np.nan,0
                                                                            ).astype(int).items()])

        repr_str += DE_status_str

        # Now consider the different filters.

        return repr_str

    def summary(self):
        print(self.__repr__())

    def get_full_table(self):
        """ Return the full input DE table along with filters created.
        """
        table_filters = self._generate_filters_table()
        return pd.concat([self.table,table_filters], axis=1)

    def _generate_filters_table(self):
        """ Generate a table of bool values with each column representing an added filter.
        """
        if len(self.gene_filters)==0:
            print("No filters found.")
            return None

        return pd.DataFrame(self.gene_filters, index=self.table.index.values)

    def get_dict_filters(self):
        """ Return a dictionary of all the added filters with associated genes lists.

        If you want to re-create a DE object from a subset of the table, this function allows you
        to directly generate a dictionary of all the available filters which were applied in the first table.

        This will map {"name_filter":[list_genes]} ; where list genes will contain all the genes from the
        original table that matched the initial filter.

        You can then easily loop through the dictionary to use <add_filtering_geneset> on the new DE_results.
        """
        tmp = self._generate_filters_table()

        filters_dict = {}
        for filter_name, bool_genes_dict in tmp.to_dict().items():
            filters_dict[filter_name] = [k for k, v in bool_genes_dict.items() if v]

        return filters_dict


    def _annotate_table_significant(self, in_place=True):
        """ Generate a bool vector basing on lFC and p value thresholds.

        If in_place, the column "significant" is added to the DE table, otherwise a vector is returned.

        The test consists in checking the p.adj to be inferior or equal to the threshold, and
        the absolute LFC to be equal or superior to the threshold.
        """
        res = self.table.apply(
                    lambda row: True if (row["padj"]<=self.thresh_pval
                                         and np.abs(row["logFC"])>=self.thresh_LFC)
                                else False,
                    axis=1)

        if in_place:
            self.table["significant"] = res
        else:
            return res

    def get_significant(self):
        """ Subset the table basing on the "significant" bool column.
        """
        return self.apply_filter("significant")

    def add_filtering_geneset(self, name_filter, list_genes):
        """ From a list of genes, store a bool vector identifying genes from the DE table.

        The "list_genes" should contain gene names which will be assigned a "True" in the
        DE table ; genes from the DE table not found in the list will be assigned a "False".

        You can reverse this True/False when applying the filter using the <apply_filter> method.
        """

        res = self.table.index.isin(list_genes)
        self.gene_filters[name_filter] = res

    def apply_filter(self, name_filter, reverse=False):
        """ Apply one of the filters previously added.
        """
        if name_filter not in self.gene_filters.keys():
            print(f"Requested filter {name_filter} not found.")
            return None

        if name_filter not in self.gene_filters:
            print("Filter not recognized.")
            return None

        if reverse:
            return self.table.loc[~self.gene_filters[name_filter],]
        else:
            return self.table.loc[self.gene_filters[name_filter],]

    def compose_filters(self, name_filter, filter_operations):
        """ Create a composite filter basing on stored filters.

        "filter_operations" should be a string, containin names of pre-existing filters,
        and composed as a query for a pandas.DataFrame table, ie:
        - "&" will get the intersection of two boolean vectors
        - "|" will get the union
        - parenthesis are accepted
        - inverting a boolean is possible through the character "~"

        Any missing filter will raise an error from Pandas, handled to return None ; in that case no filter is created.

        """
        try:
            tmp = self._generate_filters_table()
            res = tmp.query(filter_operations)
            self.gene_filters[name_filter] = self.table.index.isin(res.index.values)

        except pd.core.computation.ops.UndefinedVariableError as e:
            print(f"Filter not recognized when parsing the composite filter: {e}")
            print("No filter created.")
            return

    def _label_row_DE_status(self, row):
        """ Applies pval and LFC thresholds to identify a gene as "notSign", "belowThreshLFC", "UP", or "DOWN"

        The "noPassThreshLFC" correspond to genes that appear to be significant (from the adjusted P-value),
        but their logFC does not passes the
        """

        if row["padj"]>self.thresh_pval:
            return "notSign"

        if row["logFC"]>=self.thresh_LFC:
            return "UP"
        elif row["logFC"]<=-self.thresh_LFC:
            return "DOWN"
        else:
            return "belowThreshLFC"

        raise ValueError("Combination of pvalue and logFC ratio not recognized.")


    def _annotate_table_DE_status(self, in_place=True):
        """ Add to the DE table verbose status basing on their LFC and adjusted pvalues.

        Status are defined in the _label_row_DE_status as "notSign", "UP", or "DOWN",
        from the adjusted P-value and logFC values, using the provided thresholds.
        """
        res = self.table.apply(lambda row: self._label_row_DE_status(row), axis=1)
        if in_place:
            self.table["DE_status"] = res
        else:
            return res


    def volcano_plot(self, name_filter="significant", ax=None, show_plot=True, savefig_file=None):
        """ Plot a Volcano plot of differentialy expressed genes.
        """

        tmp = self.table.copy()

        # Here : one might want to filter out some genes that were found as
        # significant in another experiment ; so one can generate such filter, add
        # it, and then specify it to this function.
        tmp[name_filter] = self.gene_filters[name_filter]

        tmp = tmp.assign(logPvalue=-np.log10(tmp["padj"]))

        if ax is not None:
            ax1 = ax
        else:
            fig = plt.figure(figsize=(12,7))
            ax1 = fig.add_subplot(1,1,1)

        for label, label_df in tmp.groupby(name_filter):
            sns.scatterplot(data=label_df,
                            x="logFC",
                            y="logPvalue",
                            color=self.colors_filters.get(name_filter, "significant")[int(label)],
                            label=f"{name_filter} : {label} (N={label_df.shape[0]:,})",
                            ax=ax1
                           )

        ax1.set_xlabel("logFC")
        ax1.set_ylabel("-log10(adjusted P-value)")

        ax1.set_title(f"{self.condition1} vs {self.condition2}")

        ax1.get_legend().remove()
        ax1.legend(bbox_to_anchor=(1.05,1), loc='upper left')

        plt.tight_layout()

        if savefig_file:
            plt.savefig(savefig_file)

        if show_plot:
            plt.show()
        else:
            return ax1


    def logFC_cumulative_density_plot(self, name_filter, ax=None, show_plot=True, savefig_file=None):
        """ Plot genes' rank over all genes basing on their logFC, separating them using <name_filter>

        Genes in the DE table are separated using the <name_filter>.

        In both groups (<name_filter>=[True,False]), the empirical cumulative density function
        is established to associate each sorted logFC to a rank among other logFC of the group.

        These two ECDF are then plotted, and a Kolmogorov-Smirnov test is applied to evaluate
        if the two distributions are different.

        Args:
            name_filter (str) : a filter that should have been added to the list of filters.

        """
        if not name_filter in self.gene_filters:
            raise ValueError(f"{name_filter} filter not found.")

        def ecdf(data):
            """ Compute the empyrical CDF of an input array of values

            The x coordinates are the sorted values.
            the y values is the proportion of data points with a value above the considered x.

            Args:
                data (numpy.array) : array of values

            Returns:
                tuple of (x,y) coordinates.
            """
            x = np.sort(data)
            n = x.size
            y = np.arange(1, n+1) / n
            return(x,y)


        # Group of interest
        coord1 = ecdf(self.apply_filter(name_filter)["logFC"])
        # Others
        coord2 = ecdf(self.apply_filter(name_filter, reverse=True)["logFC"])

        colors = self.colors_filters.get(name_filter, ("#515050", "#d63131"))

        if ax is not None:
            ax1 = ax
        else:
            fig = plt.figure(figsize=(14,8))
            ax1 = fig.add_subplot(1,1,1)

        # We plot the "others" group first so that the group of interest lies on an upper layer.
        ax1.scatter(*coord2,
                    alpha=0.8,
                    color=colors[0],
                    label=f"{name_filter}: FALSE (N={len(coord2[0]):,})"
                   )
        ax1.scatter(*coord1,
                    alpha=0.8,
                    color=colors[1],
                    label=f"{name_filter}: TRUE (N={len(coord1[0]):,})"
                   )

        ax1.set_xlabel("logFC")
        ax1.set_ylabel("Empirical cumulative probability")

        ax1.legend(bbox_to_anchor=(1.05,1), loc='upper left')

        # Do a Kolmogorov-Smirnov test.

        # We need to check whether one of the set of coordinates is empty,
        # in which case no test can be done.
        if len(coord1[0])==0 or len(coord2[0])==0:
            ax1.set_title((f"{self.condition1} vs {self.condition2} :\nECD of log2FC "
                           f"considering the filter {name_filter}\n"
                           "(empty filter result ; no test performed)"
                          ))
        else:
            ks_test = scipy.stats.ks_2samp(coord1[0], coord2[0])

            ax1.set_title((f"{self.condition1} vs {self.condition2} :\nECD of log2FC "
                           f"considering the filter {name_filter}\n"
                           f"(Kolmogorov-Smirnov P-value: {ks_test[1]:.4})")
                          )

        plt.tight_layout()

        if savefig_file:
            plt.savefig(savefig_file)

        if show_plot:
            plt.show()
        else:
            return ax1








