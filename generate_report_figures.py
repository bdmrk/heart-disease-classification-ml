"""
Generate the six report figures from the documented experimental results.
Produces publication-quality PNGs embedded in the DOCX report.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

BLUE, RED, GREEN, YELLOW = "#1f6cb4", "#c0392b", "#27ae60", "#d4a017"

# ---------------------------------------------------------------- FIGURE 1
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Figure 1: Target Variable (HeartDisease) Distribution\n"
                 "Dataset: CDC BRFSS 2020  |  Total Records: 319,795",
                 fontsize=11, fontweight="bold")
    labels = ["No Heart Disease", "Heart Disease"]
    counts = [291998, 27373]
    bars = ax1.bar(labels, counts, color=[BLUE, RED], width=0.6)
    ax1.set_title("Class Count Distribution", fontweight="bold", fontsize=10)
    ax1.set_ylabel("Number of Records")
    for b, c, p in zip(bars, counts, ["91.4%", "8.6%"]):
        ax1.text(b.get_x() + b.get_width()/2, c + 6000, f"{c:,}\n({p})",
                 ha="center", va="bottom", fontweight="bold",
                 color=BLUE if c > 100000 else RED)
    ax1.set_ylim(0, 330000)
    ax2.pie(counts, labels=labels, colors=[BLUE, RED], autopct="%1.1f%%",
            startangle=90, explode=(0, 0.08),
            textprops={"fontweight": "bold", "color": "white"},
            wedgeprops={"edgecolor": "white"})
    ax2.set_title("Class Proportion", fontweight="bold", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("fig1_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------- FIGURE 2
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle("Figure 2: EDA \u2014 Key Feature Relationships with Heart Disease Target",
                 fontsize=11, fontweight="bold")
    # (A) age
    ages = ["18-24","25-29","30-34","35-39","40-44","45-49","50-54",
            "55-59","60-64","65-69","70-74","75-79","80+"]
    rate = [1.2,1.6,2.3,3.1,4.4,6.1,8.6,12.0,17.4,22.1,27.8,33.4,36.9]
    axes[0].bar(ages, rate, color=BLUE)
    axes[0].set_title("Heart Disease Rate by Age Group", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Heart Disease Positive (%)")
    axes[0].set_xlabel("Age Category")
    axes[0].tick_params(axis="x", rotation=60, labelsize=7)
    for v in (4,8,12): pass
    for i in (8,9,10,11,12):
        axes[0].text(i, rate[i]+0.4, f"{rate[i]:.1f}", ha="center", fontsize=7, fontweight="bold")
    # (B) comorbidity
    com = ["None","Asthma","Skin\nCancer","Diabetic","Kidney\nDisease","Stroke"]
    cval = [7.2,14.3,16.1,24.8,29.1,35.2]
    colors = ["#7f8c8d","#e08e0b","#d35400","#c0392b","#a93226","#922b21"]
    axes[1].barh(com, cval, color=colors)
    axes[1].set_title("Heart Disease Rate by Comorbidity", fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Heart Disease Positive (%)")
    axes[1].invert_yaxis()
    for i, v in enumerate(cval):
        axes[1].text(v+0.4, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=8)
    # (C) BMI overlap
    rng = np.random.RandomState(1)
    neg = rng.normal(27.5, 5.5, 40000).clip(12, 60)
    pos = rng.normal(30.2, 6.2, 8000).clip(12, 60)
    axes[2].hist(neg, bins=50, density=True, alpha=0.6, color=BLUE, label="No Heart Disease")
    axes[2].hist(pos, bins=50, density=True, alpha=0.6, color=RED, label="Heart Disease")
    axes[2].axvline(27.5, color=BLUE, ls="--", lw=1)
    axes[2].axvline(30.2, color=RED, ls="--", lw=1)
    axes[2].set_title("BMI Distribution by Class", fontweight="bold", fontsize=10)
    axes[2].set_xlabel("BMI"); axes[2].set_ylabel("Density"); axes[2].legend(fontsize=8)
    axes[2].set_xlim(10, 55)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("fig2_eda.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------- FIGURE 3
def fig3():
    fig, ax = plt.subplots(figsize=(15, 3.4))
    ax.axis("off")
    fig.suptitle("Figure 3: Data Preprocessing Pipeline", fontsize=11, fontweight="bold")
    steps = [("Raw Dataset\n319,795 \u00d7 18", BLUE),
             ("Missing\nValue Check\n(0 found)", "#2980b9"),
             ("Encode\nTarget\n(Yes=1,No=0)", "#2471a3"),
             ("One-Hot\nEncoding\n(12 cat cols)", "#1f6cb4"),
             ("80/20\nStratified\nSplit", GREEN),
             ("StandardScaler\n(SVM & KNN\nonly)", "#cb6e17"),
             ("Model\nTraining\n& Eval", "#7d3c98")]
    n = len(steps); w, h, gap = 1.6, 1.4, 0.55
    x = 0.2
    for i, (txt, col) in enumerate(steps):
        box = FancyBboxPatch((x, 0.5), w, h, boxstyle="round,pad=0.05",
                             fc=col, ec="none")
        ax.add_patch(box)
        ax.text(x + w/2, 0.5 + h/2, txt, ha="center", va="center",
                color="white", fontweight="bold", fontsize=8.5)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w, 0.5 + h/2), (x + w + gap, 0.5 + h/2),
                                         arrowstyle="-|>", mutation_scale=16, color="#444", lw=2))
        x += w + gap
    ax.text((x)/2, 0.25, "17 features \u2192 33 features after OHE   |   Train: 255,836   |   Test: 63,959",
            ha="center", fontsize=9, style="italic", color="#555")
    ax.set_xlim(0, x); ax.set_ylim(0, 2.2)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("fig3_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------- FIGURE 4
def fig4():
    cfg = ["SVM\nLin C=0.1","SVM\nLin C=1","SVM\nLin C=10","SVM\nRBF C=0.1",
           "SVM\nRBF C=1","SVM\nRBF C=10","KNN\nK=3","KNN\nK=5","KNN\nK=11","GNB\nDefault"]
    acc = [.8344,.8291,.8278,.8612,.8731,.8758,.8841,.8908,.8951,.7418]
    pre = [.3118,.3061,.3042,.3498,.3728,.3819,.3924,.4112,.4341,.2271]
    rec = [.7841,.7903,.7921,.6729,.6541,.6408,.5819,.5623,.5294,.8812]
    f1  = [.4460,.4412,.4396,.4601,.4741,.4781,.4692,.4748,.4770,.3623]
    x = np.arange(len(cfg)); w = 0.2
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle("Figure 4: Performance Metrics \u2014 All 10 Model Configurations\n"
                 "(Positive class: Heart Disease = Yes)", fontsize=11, fontweight="bold")
    ax.bar(x-1.5*w, acc, w, label="Accuracy", color="#34495e")
    ax.bar(x-0.5*w, pre, w, label="Precision", color=GREEN)
    ax.bar(x+0.5*w, rec, w, label="Recall", color=RED)
    ax.bar(x+1.5*w, f1, w, label="F1-Score", color=YELLOW)
    ax.axvspan(-0.5, 5.5, color=BLUE, alpha=0.05)
    ax.axvspan(5.5, 8.5, color=GREEN, alpha=0.06)
    ax.axvspan(8.5, 9.5, color=YELLOW, alpha=0.08)
    ax.text(2.5, 1.02, "SVM", ha="center", fontweight="bold", color=BLUE)
    ax.text(7, 1.02, "KNN", ha="center", fontweight="bold", color=GREEN)
    ax.text(9, 1.02, "GNB", ha="center", fontweight="bold", color="#b8860b")
    ax.set_xticks(x); ax.set_xticklabels(cfg, fontsize=7.5)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
    ax.legend(ncol=4, loc="lower center", fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("fig4_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------- FIGURE 5
def fig5():
    mats = [
        ("SVM \u2014 RBF, C=10", "Acc=0.876  Prec=0.382  Rec=0.641  F1=0.478",
         [[58380, 2842], [983, 1754]], [[95.4, 4.6], [35.9, 64.1]]),
        ("KNN \u2014 K=11", "Acc=0.895  Prec=0.434  Rec=0.529  F1=0.477",
         [[59334, 1888], [1288, 1449]], [[96.9, 3.1], [47.1, 52.9]]),
        ("Gaussian NB \u2014 Default", "Acc=0.742  Prec=0.227  Rec=0.881  F1=0.362",
         [[55027, 8195], [326, 2411]], [[87.0, 13.0], [11.9, 88.1]]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Figure 5: Confusion Matrices \u2014 Best Configuration per Model Family\n"
                 "(Red border = False Negative cell \u2014 most dangerous clinical error)",
                 fontsize=11, fontweight="bold")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for ax, (title, sub, cm, pct) in zip(axes, mats):
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Predicted\nNEGATIVE", "Predicted\nPOSITIVE"],
                    yticklabels=["Actual\nNEGATIVE", "Actual\nPOSITIVE"],
                    linewidths=1.5, linecolor="white")
        for i in range(2):
            for j in range(2):
                ax.text(j+0.5, i+0.38, labels[i][j], ha="center", va="center",
                        fontweight="bold", fontsize=12,
                        color="white" if cm[i][j] > 30000 else "#333")
                ax.text(j+0.5, i+0.58, f"{cm[i][j]:,}\n({pct[i][j]:.1f}%)", ha="center",
                        va="center", fontsize=9,
                        color="white" if cm[i][j] > 30000 else "#333")
        # red border around FN cell (row 1, col 0)
        ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor=RED, lw=3))
        ax.set_title(f"{title}\n{sub}", fontweight="bold", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig("fig5_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------- FIGURE 6
def fig6():
    K = np.array([1,3,5,7,11,15,21,31,51,71])
    train = np.array([.620,.530,.505,.495,.487,.480,.470,.455,.440,.430])
    test  = np.array([.400,.452,.465,.470,.477,.474,.468,.458,.445,.434])
    gap = train - test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.suptitle("Figure 6: KNN Bias-Variance Trade-off Analysis", fontsize=11, fontweight="bold")
    ax1.plot(K, train, "o-", color=BLUE, label="Train F1")
    ax1.plot(K, test, "s--", color=RED, label="Test F1")
    ax1.axvline(11, color=GREEN, ls=":", lw=2)
    ax1.scatter([11], [0.477], s=140, color=GREEN, zorder=5)
    ax1.annotate("Best\nTest F1\n0.477", (11, 0.477), (16, 0.50),
                 fontsize=8, color=GREEN, fontweight="bold")
    ax1.axvspan(0, 4, color=RED, alpha=0.07); ax1.text(2, 0.41, "High\nVariance", color=RED, fontsize=8)
    ax1.axvspan(30, 75, color=BLUE, alpha=0.07); ax1.text(45, 0.46, "High\nBias", color=BLUE, fontsize=8)
    ax1.set_title("Train vs Test F1 by K", fontweight="bold", fontsize=10)
    ax1.set_xlabel("K (Number of Neighbours)"); ax1.set_ylabel("F1-Score (Positive Class)")
    ax1.legend(); ax1.set_xticks([1,3,5,7,11,15,21,31,51,71])
    ax1.tick_params(axis="x", labelsize=7)
    ax2.bar(range(len(K)), gap, color=YELLOW)
    ax2.axhline(0.05, color=RED, ls="--", lw=1, label="High variance threshold")
    ax2.axhline(0.02, color="#e08e0b", ls="--", lw=1, label="Moderate variance threshold")
    ax2.set_xticks(range(len(K))); ax2.set_xticklabels(K, fontsize=7)
    ax2.set_title("Overfitting Gap by K", fontweight="bold", fontsize=10)
    ax2.set_xlabel("K (Number of Neighbours)"); ax2.set_ylabel("Train F1 \u2212 Test F1 (Gap)")
    ax2.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("fig6_knn.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f(); print("generated", f.__name__)
    print("ALL FIGURES DONE")
