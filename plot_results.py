import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", context="talk")
rs = np.random.RandomState(8)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_complexity():
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    bar_width = 0.35

    complexity_names = np.array(['base', '+dynamic', '+attention', '+stacked layers\n+dropout\n+init'])
    x = np.arange(len(complexity_names))

    complexity_qdmr_sari = np.array([70.6, 73.2, 77.2, 79.2])
    complexity_program_sari = np.array([74.7, 75.1, 79.6, 79.6])
    complexity_gaps_sari = complexity_program_sari - complexity_qdmr_sari

    rects1 = ax1.bar(x - bar_width/2, complexity_qdmr_sari, bar_width, label='QDMR')
    rects2 = ax1.bar(x + bar_width/2, complexity_program_sari, bar_width, label='Program')

    ax1.set_ylim(bottom=70, top=80)
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel('Absolute SARI Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(complexity_names, horizontalalignment='right')
    ax1.legend(loc=0)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # sns.set(font_scale=3)
    a = sns.barplot(x=complexity_names, y=complexity_gaps_sari, palette="rocket", ax=ax2)
    ax2.axhline(0, color="k", clip_on=False)
    ax2.tick_params(labelsize=20)
    ax2.set_ylabel("Gaps (Program - QDMR)")

    sns.despine(bottom=True)
    plt.tight_layout(h_pad=2)
    plt.show()


def plot_capacity(qdmr, minimized, program, plot_name):
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle(plot_name, fontsize=16)
    bar_width = 0.1

    capacities = [16,64,256,512,1024]
    # Set position of bar on X axis
    r1 = np.arange(len(capacities))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]



    program_minimized_gap = program - minimized
    minimized_qdmr_gap = minimized - qdmr
    program_qdmr_gap = program - qdmr

    min_y = min(min(qdmr), min(minimized), min(program)) - 2
    print(min_y)


    rects1 = ax1.bar(r1, qdmr, bar_width, label='QDMR')
    rects2 = ax1.bar(r2, minimized, bar_width, label='Minimized Program')
    rects3 = ax1.bar(r3, program, bar_width, label='Program')

    ax1.set_ylim(bottom=min_y, top=80)
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel('Absolute SARI Scores')
    ax1.set_xticks(r2)
    ax1.set_xticklabels(capacities, horizontalalignment='right')
    ax1.legend(loc=1)

    # autolabel(rects1, ax1)
    # autolabel(rects2, ax1)
    # autolabel(rects3, ax1)

    # -----------------------GAPS ------------------------------------------

    rects1 = ax2.bar(r1, minimized_qdmr_gap, bar_width, label='Minimized - QDMR')
    rects2 = ax2.bar(r2, program_minimized_gap, bar_width, label='Program - Minimized')
    rects3 = ax2.bar(r3, program_qdmr_gap, bar_width, label='Program - QDMR')

    # ax1.set_ylim(bottom=70, top=80)
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set_ylabel('Realtive SARI Scores')
    ax2.set_xticks(r2)
    ax2.set_xticklabels(capacities, horizontalalignment='right')
    ax2.legend(loc=1)

    # autolabel(rects1, ax2)
    # autolabel(rects2, ax2)
    # autolabel(rects3, ax2)

    plt.tight_layout()
    plt.show()


# plot_complexity()
# ----- Capacity-------------
vanilla_qdmr = np.array([71.1, 76.9, 79.3, 77.9, 76.2])
vanilla_minimized = np.array([71.2, 74.7, 75.6, 79.1, 76.3])
vanilla_program = np.array([70.1, 72.7, 79.6, 75.5, 75.7])
plot_capacity(vanilla_qdmr, vanilla_minimized, vanilla_program, 'Capacity - Vanilla Split')


domain_qdmr = np.array([50.7, 52.9, 52.1, 52.9, 52])
domain_minimized = np.array([63.7, 65.1, 65.5, 65.5, 65.5])
domain_program = np.array([63, 64.5, 65.8, 65.6, 65.7])
plot_capacity(domain_qdmr, domain_minimized, domain_program, 'Capacity - Domain Split')


length_qdmr = np.array([54.7, 56.6, 56.7, 56, 56.4])
length_minimized = np.array([73.7, 74.8, 75.8, 75.9, 75.1])
length_program = np.array([73.4, 75, 75.5, 75.5, 75.4])
plot_capacity(length_qdmr, length_minimized, length_program, 'Capacity - Length Split')


