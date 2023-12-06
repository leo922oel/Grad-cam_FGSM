import matplotlib.pyplot as plt

def Comp_Full_Iter(examples, eps_list, num_iter, fname='comp_full_iter.jpg'):
    print("===== Output Full & Iteration Comp. =====")
    count = 0
    plt.figure(figsize=(20, 11))
    row = len(examples[0]) - 1
    for i , eps in enumerate(eps_list):
        for idx, (pred, perturbed, heatmap1, heatmap2) in enumerate(examples):
            if (idx not in [0, 1, len(examples)-1]):
                continue
            count += 1
            plt.subplot(row, 3, count)
            plt.xticks([], [])
            plt.yticks([], [])
            if idx == 0:
                plt.ylabel(f"Original", fontsize=30)
            elif idx == 1:
                plt.ylabel(f"Epslion: {eps}", fontsize=30)
                plt.title(f"FGSM\npred: {pred}", fontsize=40)
            else:
                plt.title(f"iFGSM\npred: {pred}", fontsize=40)
                plt.ylabel(f"# of iter: {num_iter}", fontsize=30)
            plt.imshow(perturbed, cmap='gray')

            plt.subplot(row, 3, count + 3)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f"")
            plt.imshow(heatmap1, )

            plt.subplot(row, 3, count + 3*2)
            # heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_RGB2GRAY)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f"")
            plt.imshow(heatmap2)

    plt.tight_layout()
    plt.savefig(fname)
    print("===== Finish =====")

def show_iter(examples, eps_list, fname='show_iter.jpg'):
    print("===== Output Iteration display =====")
    count = 0
    plt.figure(figsize=(20, 11))
    row = len(examples[0]) - 1
    num_iter = len(examples)
    for i , eps in enumerate(eps_list):
        for idx, (pred, perturbed, heatmap1, heatmap2) in enumerate(examples):
            count += 1
            plt.subplot(row, num_iter, count)
            plt.xticks([], [])
            plt.yticks([], [])
            if idx == 0:
                plt.ylabel(f"Original", fontsize=30)
            elif idx == 1:
                plt.ylabel(f"Epslion: {eps}", fontsize=30)
            plt.title(f"pred: {pred}", fontsize=40)
            plt.imshow(perturbed, cmap='gray')

            plt.subplot(row, num_iter, count + num_iter)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f"")
            plt.imshow(heatmap1, )

            plt.subplot(row, num_iter, count + num_iter*2)
            # heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_RGB2GRAY)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f"")
            plt.imshow(heatmap2)

    plt.tight_layout()
    plt.savefig(fname)
    print("===== Finish =====")

def noise_plot(examples, noises, fname='noises_plot.jpg'):
    print("===== Output analysis figure =====")
    count = 0
    plt.figure(figsize=(20, 11))
    row = len(examples[0]) - 1
    num_iter = len(examples)
    for i , eps in enumerate(eps_list):
        for idx, (pred, perturbed, heatmap1, heatmap2) in enumerate(examples):
            if (idx not in [0, 1, len(examples)-1]):
                continue
            count += 1
            plt.subplot(row, 3, count)
            plt.xticks([], [])
            plt.yticks([], [])
            if idx == 0:
                plt.ylabel(f"Original", fontsize=30)
            elif idx == 1:
                plt.ylabel(f"Epslion: {eps}", fontsize=30)
            plt.imshow(perturbed, cmap='gray')

            plt.subplot(row, 3, count + num_iter)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f"")
            plt.imshow(heatmap1, )

            plt.subplot(row, 3, count + num_iter*2)
            # heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_RGB2GRAY)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f"")
            plt.imshow(heatmap2)

    plt.tight_layout()
    plt.savefig(fname)
    print("===== Finish =====")