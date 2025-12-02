import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
plt.rcParams['font.weight'] = 'bold'
import seaborn as sns
import os
# configs
sns.set_theme(style="ticks")

import numpy as np
import json

PATH_RESULTS = "./documents/local_data/results/"
PATH_PLOTS = "./documents/local_data/visualizations/"

tasks = ["SICAPv2", "Skin", "NCT", "CCRCC", "MESSIDOR", "FIVES", "MMAC", "mBRSET", "CheXpert", "NIH", "COVID", "PadChest"]

color_mapping = {
    "simpleshot": "darkred",
    "zslp":       "lightblue",
    "clap":       "dodgerblue",
    "lp_pp":      "darkblue",
    "sstext":     "lightsalmon",
    "sstext_p":   "orange",
    "sstext_u":   "gold",
}

running_times = {
    "zslp":       {1: 0.2245, 2: 0.1626, 4: 0.2667, 8: 0.2940, 16: 0.2898},
    "clap":       {1: 0.4946, 2: 0.466, 4: 0.5457, 8: 0.4686, 16: 0.5469},
    "lp_pp":      {1: 0.68  , 2: 0.6387, 4: 0.6677, 8: 0.6752, 16: 1.0352},
    "sstext_p":   {1: 0.0007, 2: 0.0007, 4: 0.0009, 8: 0.0008, 16: 0.0009},
    "sstext_u":   {1: 0.002,  2: 0.0047, 4: 0.0032, 8: 0.0032, 16: 0.003},
}

classes = {
    "SICAPv2": 4, "Skin": 16, "NCT": 9, "CCRCC": 4,
    "MESSIDOR": 5, "FIVES": 4, "MMAC": 5, "mBRSET": 5,
    "CheXpert": 5, "NIH": 19, "COVID": 4, "PadChest": 5
}

entropy = {
    "SICAPv2": 0.93, "Skin": 0.89, "NCT": 0.96, "CCRCC": 0.83, 
    "MESSIDOR": 0.71, "FIVES": 1.0, "MMAC": 0.82, "mBRSET": 0.50, 
    "CheXpert": 1.0, "NIH": 1.0, "COVID": 0.94, "PadChest": 0.67
}

silh_score = {
    "SICAPv2": 0.1217, "Skin": 0.3702, "NCT": 0.4219, "CCRCC": 0.2328, 
    "MESSIDOR": 0.1631, "FIVES": 0.1739, "MMAC": 0.0303, "mBRSET": 1e-04, 
    "CheXpert": -0.0038, "NIH": -0.0669, "COVID": 0.118, "PadChest": 0.1217
}

def load_json(path):
    json1_file = open(path)
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)
    return json1_data

results_zslp           = load_json(PATH_RESULTS + "LP_main_11272031")
results_lp_pp          = load_json(PATH_RESULTS + "LP++_main_11272115")
results_clap           = load_json(PATH_RESULTS + "CLAP_main_11272054")
results_ss             = load_json(PATH_RESULTS + "SS_main_11272010")
results_sstext         = load_json(PATH_RESULTS + "SS-Text_main_11272012")
results_sstext_p_repel = load_json(PATH_RESULTS + "SS-Text+_main_11272015")
results_sstext_u       = load_json(PATH_RESULTS + "SS-Text-U_main_11272006")

# FEW-SHOT PLOTS FUNCTION
def fs_plot(res_list, labels, colors, title=None, image_name="no_title", target_variable="aca", axis_label_size=22, legend_titles_size=18, shots=[1, 2, 4, 8, 16], ylims=[44, 67],
            yticks=[45, 50, 55, 60, 65, 70, 75], subfolder_name="",classes=None, H=None, ncols=1, S=None):

    yaxis_label = {"acc": "Accuracy", "aca": "ACA"}

    for i in range(len(res_list)):
        plt.plot(shots, [res_list[i][target_variable][str(shot)] for shot in shots], linestyle='-', marker='o', color=colors[i], alpha=0.4, markersize=12, linewidth=4, markeredgecolor='k')

    # Legends and axis
    if labels is not None:
        plt.legend(labels=labels, loc='lower right', prop={'weight': 'bold', 'size': legend_titles_size}, framealpha=1, ncols=ncols, handletextpad=0.4, columnspacing=0.4, handlelength=1.0)

    for i in range(len(res_list)):
        plt.plot(shots, [res_list[i][target_variable][str(shot)] for shot in shots], linestyle='none', marker='o', color=colors[i], alpha=1, markersize=12, linewidth=4, markeredgecolor='k', markeredgewidth=1.5)

    if title is not None:
        plt.title(title, fontsize=axis_label_size, weight="bold")
    plt.xlabel(r"Support set (K)", fontsize=axis_label_size, weight="bold")
    plt.ylabel(yaxis_label[target_variable], fontsize=axis_label_size, weight="bold")
    plt.tick_params(axis='both', which='major', labelsize=axis_label_size-2)
    ax = plt.gca()
    plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-1)
    plt.yticks(yticks, yticks)
    plt.xticks([0,1,2,4,8,16],[0,1,2,4,8,16])
    ax = plt.gca().axis()

    plt.plot(0, res_list[0][target_variable]["0"], linestyle='none', marker='*', color="darkblue", alpha=1, markersize=15, linewidth=5, markeredgecolor='k', markeredgewidth=1.5)

    plt.gca().axis((-0.5, shots[-1]+1, ylims[0], ylims[1]))
    ax = plt.gca()
    for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
    plt.tight_layout()

    # Plot number of classes and label-marginal entropy
    if classes is not None:
        text = "" + r"$C=" + str(classes) + " \ | \ H(Y)=" + str(H) + "\ | \ S=" + str(S) + "$"
        ax.text(0, ylims[1]-(ylims[1]-ylims[0])*0.08, text, fontsize=16, color='black',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    if not os.path.exists(PATH_PLOTS + subfolder_name):
            os.makedirs(PATH_PLOTS + subfolder_name)
    path = PATH_PLOTS + subfolder_name + image_name + "_" + target_variable + ".png"
    plt.savefig(path, dpi=200, format='png', bbox_inches='tight')
    plt.close()

# MAIN PLOT - AVERAGE FOR ALL TASKS

fs_plot(res_list=[results_zslp["average"], results_lp_pp["average"], results_clap["average"], results_sstext["average"], results_sstext_p_repel["average"], results_sstext_u["average"]],
        labels=[r"ZSLP$_{\text{[CVPR'24]}}$", r"${\text{LP}}^{++}_{\text{[CVPR'24]}}$", r"CLAP$_{\text{[CVPR'24]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["zslp"], color_mapping["lp_pp"], color_mapping["clap"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="Average 12 datasets", image_name="main_results", target_variable="aca", subfolder_name="main_results/", legend_titles_size=15, ylims=[38, 67],
        yticks=[40, 45, 50, 55, 60, 65, 70, 75], ncols=2)


# PLOTS PER EACH DATASET

fs_plot(res_list=[results_ss["SICAPv2"],
                  results_sstext["SICAPv2"],
                  results_sstext_p_repel["SICAPv2"],
                  results_sstext_u["SICAPv2"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="SICAPv2", image_name="SICAPv2", target_variable="aca", ylims=[33, 78], yticks=[35, 40, 45, 50, 55, 60, 65, 70, 75],
        classes=classes["SICAPv2"], H=entropy["SICAPv2"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["SICAPv2"], 2))

fs_plot(res_list=[results_ss["Skin"],
                  results_sstext["Skin"],
                  results_sstext_p_repel["Skin"],
                  results_sstext_u["Skin"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="Skin", image_name="Skin", target_variable="aca", ylims=[43, 91], yticks=[45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        classes=classes["Skin"], H=entropy["Skin"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["Skin"], 2))

fs_plot(res_list=[results_ss["NCT"],
                  results_sstext["NCT"],
                  results_sstext_p_repel["NCT"],
                  results_sstext_u["NCT"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="NCT", image_name="NCT", target_variable="aca", ylims=[58, 97], yticks=[60, 65, 70, 75, 80, 85, 90, 95],
        classes=classes["NCT"], H=entropy["NCT"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["NCT"], 2))

fs_plot(res_list=[results_ss["CCRCC"],
                  results_sstext["CCRCC"],
                  results_sstext_p_repel["CCRCC"],
                  results_sstext_u["CCRCC"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="CCRCC", image_name="CCRCC", target_variable="aca", ylims=[50, 83], yticks=[55, 60, 65, 70, 75, 80],
        classes=classes["CCRCC"], H=entropy["CCRCC"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["CCRCC"], 2))

fs_plot(res_list=[results_ss["MESSIDOR"],
                  results_sstext["MESSIDOR"],
                  results_sstext_p_repel["MESSIDOR"],
                  results_sstext_u["MESSIDOR"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="MESSIDOR", image_name="MESSIDOR", target_variable="aca", ylims=[29, 78], yticks=[30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        classes=classes["MESSIDOR"], H=entropy["MESSIDOR"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["MESSIDOR"], 2))

fs_plot(res_list=[results_ss["FIVES"],
                  results_sstext["FIVES"],
                  results_sstext_p_repel["FIVES"],
                  results_sstext_u["FIVES"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="FIVES", image_name="FIVES", target_variable="aca", ylims=[56, 77], yticks=[60, 65, 70, 75],
        classes=classes["FIVES"], H=entropy["FIVES"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["FIVES"], 2))

fs_plot(res_list=[results_ss["MMAC"],
                  results_sstext["MMAC"],
                  results_sstext_p_repel["MMAC"],
                  results_sstext_u["MMAC"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="MMAC", image_name="MMAC", target_variable="aca", ylims=[27, 57], yticks=[30, 35, 40, 45, 50, 55],
        classes=classes["MMAC"], H=entropy["MMAC"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["MMAC"], 2))

fs_plot(res_list=[results_ss["mBRSET"],
                  results_sstext["mBRSET"],
                  results_sstext_p_repel["mBRSET"],
                  results_sstext_u["mBRSET"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="mBRSET", image_name="mBRSET", target_variable="aca", ylims=[24, 54], yticks=[25, 30, 35, 40, 45, 50],
        classes=classes["mBRSET"], H=entropy["mBRSET"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["mBRSET"], 2))
fs_plot(res_list=[results_ss["CheXpert"],
                  results_sstext["CheXpert"],
                  results_sstext_p_repel["CheXpert"],
                  results_sstext_u["CheXpert"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="CheXpert", image_name="CheXpert", target_variable="aca", ylims=[31, 54], yticks=[35, 40, 45, 50],
        classes=classes["CheXpert"], H=entropy["CheXpert"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["CheXpert"], 2))

fs_plot(res_list=[results_ss["NIH"],
                  results_sstext["NIH"],
                  results_sstext_p_repel["NIH"],
                  results_sstext_u["NIH"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="NIH", image_name="NIH", target_variable="aca", ylims=[9, 36], yticks=[10, 15, 20, 25, 30, 35],
        classes=classes["NIH"], H=entropy["NIH"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["NIH"], 2))

fs_plot(res_list=[results_ss["COVID"],
                  results_sstext["COVID"],
                  results_sstext_p_repel["COVID"],
                  results_sstext_u["COVID"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="COVID", image_name="COVID", target_variable="aca", ylims=[18, 76], yticks=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        classes=classes["COVID"], H=entropy["COVID"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["COVID"], 2))

fs_plot(res_list=[results_ss["PadChest"],
                  results_sstext["PadChest"],
                  results_sstext_p_repel["PadChest"],
                  results_sstext_u["PadChest"]],
        labels=[r"SS$_{\text{[ArXiv'19]}}$",r"${\text{SS-Text}}_{\text{[IPMI'25]}}$",r"${\text{SS-Text}}^{+}_{\text{[MICCAI'25]}}$",r"SS-Text-U$_{\text{(Ours)}}$"],
        colors=[color_mapping["simpleshot"], color_mapping["sstext"], color_mapping["sstext_p"], color_mapping["sstext_u"]],
        title="PadChest", image_name="PadChest", target_variable="aca",
        ylims=[32, 69], yticks=[35, 40, 45, 50, 55, 60, 65],
        classes=classes["PadChest"], H=entropy["PadChest"], subfolder_name="results_per_dataset/", legend_titles_size=18, S=np.round(silh_score["PadChest"], 2))


## COMPUTATIONAL EFFICIENCY

shots = [2, 4, 8]
smax = 600

plt.scatter([np.log10(1/running_times["zslp"][shot]) for shot in shots][::-1],
            [results_zslp["average"]["aca"][str(shot)] for shot in shots][::-1], marker='o', color=color_mapping["zslp"], alpha=1.0,
            s=list(smax * np.array(shots) / 16)[::-1], edgecolors='k', linewidth=2, zorder=3)

plt.scatter([np.log10(1/running_times["clap"][shot]) for shot in shots][::-1],
            [results_clap["average"]["aca"][str(shot)] for shot in shots][::-1], marker='o', color=color_mapping["clap"], alpha=1.0,
            s=list(smax * np.array(shots) / 16)[::-1], edgecolors='k', linewidth=2, zorder=3)

plt.scatter([np.log10(1/running_times["lp_pp"][shot]) for shot in shots][::-1],
            [results_lp_pp["average"]["aca"][str(shot)] for shot in shots][::-1], marker='o', color=color_mapping["lp_pp"], alpha=1.0,
            s=list(smax * np.array(shots) / 16)[::-1], edgecolors='k', linewidth=2, zorder=3)

plt.scatter([np.log10(1/running_times["sstext_p"][shot]) for shot in shots][::-1],
            [results_sstext_p_repel["average"]["aca"][str(shot)] for shot in shots][::-1], marker='o', color=color_mapping["sstext_p"], alpha=1.0,
            s=list(smax * np.array(shots) / 16)[::-1], edgecolors='k', linewidth=2, zorder=3)

plt.scatter([np.log10(1/running_times["sstext_u"][shot]) for shot in shots][::-1],
            [results_sstext_u["average"]["aca"][str(shot)] for shot in shots][::-1], marker='o', color=color_mapping["sstext_u"], alpha=1.0,
            s=list(smax * np.array(shots) / 16)[::-1], edgecolors='k', linewidth=2, zorder=3)

# Legends and axis
plt.legend(labels=["ZSLP", "CLAP", "LP++", "SS-Text+", "SS-Text-U"], loc='upper center',
           prop={'weight': 'bold', 'size': 17},
           framealpha=1, ncols=3, handletextpad=0.02, columnspacing=0.1).set_zorder(1)

plt.plot([np.log10(1/running_times["zslp"][shot]) for shot in shots][::-1],
         [results_zslp["average"]["aca"][str(shot)] for shot in shots][::-1], linestyle='-', marker='none', color=color_mapping["zslp"], alpha=0.3,
         markersize=14, linewidth=7, markeredgecolor='k', zorder=2)

plt.plot([np.log10(1/running_times["clap"][shot]) for shot in shots][::-1],
            [results_clap["average"]["aca"][str(shot)] for shot in shots][::-1], linestyle='-', marker='none', color=color_mapping["clap"], alpha=0.3,
         markersize=14, linewidth=7, markeredgecolor='k', zorder=2)

plt.plot([np.log10(1/running_times["lp_pp"][shot]) for shot in shots][::-1],
            [results_lp_pp["average"]["aca"][str(shot)] for shot in shots][::-1], linestyle='-', marker='none', color=color_mapping["lp_pp"], alpha=0.3,
         markersize=14, linewidth=7, markeredgecolor='k', zorder=2)

plt.plot([np.log10(1/running_times["sstext_p"][shot]) for shot in shots][::-1],
            [results_sstext_p_repel["average"]["aca"][str(shot)] for shot in shots][::-1], linestyle='-', marker='none', color=color_mapping["sstext_p"], alpha=0.3,
         markersize=14, linewidth=7, markeredgecolor='k', zorder=2)

plt.plot([np.log10(1/running_times["sstext_u"][shot]) for shot in shots][::-1],
            [results_sstext_u["average"]["aca"][str(shot)] for shot in shots][::-1], linestyle='-', marker='none', color=color_mapping["sstext_u"], alpha=0.3,
         markersize=14, linewidth=7, markeredgecolor='k', zorder=2)

plt.xlabel(r"log$_{10}$ fits/second", fontsize=24, weight="bold")
plt.ylabel('ACA', fontsize=24, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=22)
ax = plt.gca()
ax.text(0.87, 54, r"$K=2$", fontsize=13, color='black')
ax.text(0.63, 58.5, r"$K=4$", fontsize=13, color='black')
ax.text(0.62, 61, r"$K=8$", fontsize=13, color='black')
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)
plt.yticks([45, 50, 55, 60, 65, 70],[45, 50, 55, 60, 65, 70])
plt.xticks([0,1.0,2.0, 3.0],[0,1.0,2.0, 3.0])
ax = plt.gca().axis()

plt.gca().axis((0., 3.5, 52, 68))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()

path = PATH_PLOTS +  "main_results/efficiency.png"
plt.savefig(path, dpi=200, format='png', bbox_inches='tight')
plt.close()


## EXPLORATORY ANALYSIS VIA SHILHOUETT SCORE

silh =  [silh_score[itask]for itask in tasks]
zero_shot =  [results_sstext_u[itask]["aca"]["0"] for itask in tasks]
low_shot =  [results_sstext_u[itask]["aca"]["1"] for itask in tasks]
high_shot =  [results_sstext_u[itask]["aca"]["16"] for itask in tasks]

rho_zs = np.corrcoef(np.array(silh), np.array(zero_shot))[1,1]
rho_ls = np.corrcoef(np.array(silh), np.array(low_shot))[1,1]
rho_hs = np.corrcoef(np.array(silh), np.array(high_shot))[1,1]

m_zs,b_zs = np.polyfit(silh, zero_shot, 1)
m_ls,b_ls = np.polyfit(silh, low_shot, 1)
m_hs,b_hs = np.polyfit(silh, high_shot, 1)

coeffs = np.polyfit(silh,low_shot,deg=1)
poly_ls = np.poly1d(coeffs)
yfit_ls = lambda x: poly_ls(x)

coeffs = np.polyfit(silh,high_shot,deg=1)
poly_hs = np.poly1d(coeffs)
yfit_hs = lambda x: poly_hs(x)

coeffs = np.polyfit(silh,zero_shot,deg=2)
poly = np.poly1d(coeffs)
yfit_zs = lambda x: poly(x)

plt.plot(silh, high_shot, linestyle='none', marker='o', color="blueviolet", alpha=0.8, markersize=20, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(silh, low_shot, linestyle='none', marker='o', color="thistle", alpha=0.8, markersize=16, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(silh, zero_shot, linestyle='none', marker='*', color="darkblue", alpha=1, markersize=12, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

# Legends and axis
plt.legend(labels=[r"16-shot-$\rho=0.94$", r"  1-shot-$\rho=0.95$", r"  0-shot-$\rho=0.65$"], loc='upper left', prop={'weight': 'bold', 'size': 12}, framealpha=1)

xline = np.arange(-0.1, 0.42, step=0.01)
plt.plot(xline, [yfit_ls(x) for x in xline], linestyle="--", color="thistle")
plt.plot(xline, [yfit_hs(x) for x in xline], linestyle="--", color="blueviolet")
plt.plot(xline, [yfit_zs(x) for x in xline], linestyle="--", color="darkblue")

plt.xlabel(r"Dataset Silhouette score (w/ labels)", fontsize=20, weight="bold")
plt.ylabel('ACA', fontsize=20, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=18)
ax = plt.gca()
plt.yticks([25, 35, 45, 55, 65, 75, 85, 95],[25, 35, 45, 55, 65, 75, 85, 95])
plt.xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],[-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax = plt.gca().axis()
plt.gca().axis((-0.2, +0.5, 18, 102))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

path = PATH_PLOTS +  "main_results/Silhouette.png"
plt.savefig(path, dpi=200, format='png', bbox_inches='tight')
plt.close()


# EFFECT OF M - NUMBER OF SHOTS FOR UNLABELED DATA


results_sstext_u_0MMiter    = load_json(PATH_RESULTS + "SS-Text-U_M0_11281321")
results_sstext_u_MK1        = load_json(PATH_RESULTS + "SS-Text-U_M1_11281301")
results_sstext_u_MK2        = load_json(PATH_RESULTS + "SS-Text-U_M2_11281305")
results_sstext_u_MK4        = load_json(PATH_RESULTS + "SS-Text-U_M4_11281307")
results_sstext_u_MK8        = load_json(PATH_RESULTS + "SS-Text-U_M8_11281310")
results_sstext_u_MK12       = load_json(PATH_RESULTS + "SS-Text-U_M12_11281312")
results_sstext_u_MK16       = load_json(PATH_RESULTS + "SS-Text-U_M16_11281314")
results_sstext_u_MK20       = load_json(PATH_RESULTS + "SS-Text-U_M20_11281317")
results_sstext_u            = load_json(PATH_RESULTS + "SS-Text-U_main_11272006")

M = [0,1,2,4,8,12,16,20,24]
results_raw = [results_sstext_u_0MMiter, results_sstext_u_MK1, results_sstext_u_MK2,
               results_sstext_u_MK4, results_sstext_u_MK8, results_sstext_u_MK12,
               results_sstext_u_MK16, results_sstext_u_MK20, results_sstext_u]

results = {1: [], 2: [], 4: []}
for i in [1, 2, 4]:
    for res in results_raw:\
        results[i].append(res["average"]["aca"][str(i)])


plt.plot(M, results[1], linestyle='-', marker='s', color="thistle", alpha=0.4, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(M, results[2], linestyle='-', marker='s', color="palevioletred", alpha=0.3, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(M, results[4], linestyle='-', marker='s', color="darkviolet", alpha=0.3, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)

# Legends and axis
plt.legend(labels=["1-shot", "2-shot", "4-shot"], loc='lower right', prop={'weight': 'bold', 'size': 20}, framealpha=1)

plt.plot(M, results[1], linestyle='none', marker='s', color="thistle", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(M, results[2], linestyle='none', marker='s', color="palevioletred", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(M, results[4], linestyle='none', marker='s', color="darkviolet", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.xlabel(r"Unlabeled data M (shots)", fontsize=24, weight="bold")
plt.ylabel('ACA', fontsize=24, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=22)
ax = plt.gca()
plt.yticks([45, 50, 55, 60],[45, 50, 55, 60])
plt.xticks([0,2,4,8,12,16,20,24],[0,2,4,8,12,16,20,24])
ax = plt.gca().axis()
plt.gca().axis((-1, 25, 42, 63))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

if not os.path.exists(PATH_PLOTS +  "ablation_studies/"):
        os.makedirs(PATH_PLOTS +  "ablation_studies/")
path = PATH_PLOTS +  "ablation_studies/unlabeled_data.png"
plt.savefig(path)
plt.close()


# EFFECT OF ITERATIONS IN MM

results_sstext_u_0MMiter = load_json(PATH_RESULTS + "SS-Text-U_M0_11281321")
results_sstext_u_1MMiter = load_json(PATH_RESULTS + "SS-Text-U_MMiters1_11281332")
results_sstext_u_2MMiter = load_json(PATH_RESULTS + "SS-Text-U_MMiters2_11281335")
results_sstext_u_3MMiter = load_json(PATH_RESULTS + "SS-Text-U_main_11272006")
results_sstext_u_4MMiter = load_json(PATH_RESULTS + "SS-Text-U_MMiters4_11281339")
results_sstext_u_5MMiter = load_json(PATH_RESULTS + "SS-Text-U_MMiters5_11281341")


reps = [0, 1, 2, 3, 4, 5]
results_raw = [results_sstext_u_0MMiter, results_sstext_u_1MMiter, results_sstext_u_2MMiter,
               results_sstext_u_3MMiter, results_sstext_u_4MMiter, results_sstext_u_5MMiter]

results = {1: [], 2: [], 4: []}
for i in [1, 2, 4]:
    for res in results_raw:\
        results[i].append(res["average"]["aca"][str(i)])


plt.plot(reps, results[1], linestyle='-', marker='s', color="thistle", alpha=0.4, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(reps, results[2], linestyle='-', marker='s', color="palevioletred", alpha=0.3, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(reps, results[4], linestyle='-', marker='s', color="darkviolet", alpha=0.3, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)

# Legends and axis
plt.legend(labels=["1-shot", "2-shot", "4-shot"], loc='lower right', prop={'weight': 'bold', 'size': 20}, framealpha=1)

plt.plot(reps, results[1], linestyle='none', marker='s', color="thistle", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(reps, results[2], linestyle='none', marker='s', color="palevioletred", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(reps, results[4], linestyle='none', marker='s', color="darkviolet", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.xlabel(r"T(iterations)", fontsize=24, weight="bold")
plt.ylabel('ACA', fontsize=24, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=22)
ax = plt.gca()
plt.yticks([45, 50, 55, 60],[45, 50, 55, 60])
plt.xticks([0,1,2,3,4,5],[0,1,2,3,4,5])
ax = plt.gca().axis()
plt.gca().axis((-0.3, 5.6, 42, 63))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()

plt.plot([3, 3], [0, 63], linestyle='--', marker='none', color="red", linewidth=3, alpha=0.4, zorder=2)
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

path = PATH_PLOTS +  "ablation_studies/reps_mm.png"
plt.savefig(path)
plt.close()


# EFFECT OF ITERATIONS IN Optimal Transport

results_sstext_u_0OTiter  = load_json(PATH_RESULTS + "SS-Text-U_OTiters0_11281343")
results_sstext_u_1OTiter  = load_json(PATH_RESULTS + "SS-Text-U_OTiters1_11281345")
results_sstext_u_2OTiter  = load_json(PATH_RESULTS + "SS-Text-U_OTiters2_11281348")
results_sstext_u_4OTiter  = load_json(PATH_RESULTS + "SS-Text-U_OTiters4_11281353")
results_sstext_u_6OTiter  = load_json(PATH_RESULTS + "SS-Text-U_OTiters6_11281355")
results_sstext_u_8OTiter  = load_json(PATH_RESULTS + "SS-Text-U_OTiters8_11281400")
results_sstext_u_10OTiter = load_json(PATH_RESULTS + "SS-Text-U_main_11272006")
results_sstext_u_12OTiter = load_json(PATH_RESULTS + "SS-Text-U_OTiters12_11281402")

reps = [0,1,2,4,6,8,10,12]
results_raw = [results_sstext_u_0OTiter, results_sstext_u_1OTiter, results_sstext_u_2OTiter,
               results_sstext_u_4OTiter, results_sstext_u_6OTiter, results_sstext_u_8OTiter,
               results_sstext_u_10OTiter, results_sstext_u_12OTiter]

results = {1: [], 2: [], 4: []}
for i in [1, 2, 4]:
    for res in results_raw:\
        results[i].append(res["average"]["aca"][str(i)])


plt.plot(reps, results[1], linestyle='-', marker='s', color="thistle", alpha=0.4, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(reps, results[2], linestyle='-', marker='s', color="palevioletred", alpha=0.3, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(reps, results[4], linestyle='-', marker='s', color="darkviolet", alpha=0.3, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)

# Legends and axis
plt.legend(labels=["1-shot", "2-shot", "4-shot"], loc='lower center', prop={'weight': 'bold', 'size': 20}, framealpha=1)

plt.plot(reps, results[1], linestyle='none', marker='s', color="thistle", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(reps, results[2], linestyle='none', marker='s', color="palevioletred", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(reps, results[4], linestyle='none', marker='s', color="darkviolet", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.xlabel(r"$\text{T}_{\text{OT}}\text{(iterations)}$", fontsize=24, weight="bold")
plt.ylabel('ACA', fontsize=24, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=22)
ax = plt.gca()
plt.yticks([45, 50, 55, 60],[45, 50, 55, 60])
plt.xticks([0,1,2,4,6,8,10,12],[0,1,2,4,6,8,10,12])
ax = plt.gca().axis()
plt.gca().axis((-1, 13, 42, 63))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()

plt.plot([10, 10], [0, 63], linestyle='--', marker='none', color="red", linewidth=3, alpha=0.4, zorder=2)
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

path = PATH_PLOTS +  "ablation_studies/reps_ot.png"
plt.savefig(path)
plt.close()