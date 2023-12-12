import matplotlib.pyplot as plt

def Comp_Full_Iter(example, eps, num_iter, fname='comp_full_iter.jpg'):
    print("===== Output Full & Iteration Comp. =====")
    count = 0
    plt.figure(figsize=(20, 11))
    row = len(example[0]) - 2
    for idx, (pred, perturbed, heatmap1, heatmap2, _) in enumerate(example):
        if (idx not in [0, 1, len(example)-1]):
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
        plt.imshow(heatmap1)

        plt.subplot(row, 3, count + 3*2)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(f"")
        plt.imshow(heatmap2)

    plt.tight_layout()
    plt.savefig(fname)
    print("===== Finish =====")

def show_iter(example, eps, fname='show_iter.jpg'):
    print("===== Output Iteration display =====")
    count = 0
    plt.figure(figsize=(20, 11))
    row = len(example[0]) - 2
    num_iter = min(len(example), 10)
    for idx, (pred, perturbed, heatmap1, heatmap2, _) in enumerate(example):
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

def show_mask(examples, fname="Mask_demon.jpg"):
    print("===== Output mask demonstration =====")
    count = 0
    plt.figure(figsize=(20, 11))
    row = len(examples[0])-2
    num_iter = len(examples)
    num_iter = min(num_iter, 10)
    for idx, (_, perturbed, heatmap1, _, mask) in enumerate(examples):
        if idx >= 10: break
        count += 1
        plt.subplot(row, num_iter, count)
        plt.xticks([], [])
        plt.yticks([], [])
        if idx == 0:
            plt.ylabel(f"Perturbed Iamge", fontsize=30)
        plt.imshow(perturbed, cmap='gray')

        plt.subplot(row, num_iter, count + num_iter*1)
        if idx == 0:
            plt.ylabel("Grad CAM", fontsize=30)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(f"")
        plt.imshow(heatmap1, )
        # if idx == len(examples)-1:
            # plt.colorbar()

        plt.subplot(row, num_iter, count + num_iter*2)
        if idx == 0:
            plt.ylabel("Mask", fontsize=30)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(f"")
        plt.imshow(mask, cmap='gray')

    plt.tight_layout()
    plt.savefig(fname)
    print("===== Finish =====")

def box(eps_list, noise_list, fname="box.jpg"):
    print("===== Output Full & Iteration Comp. =====")
    # print(noise_list)
    plt.figure(figsize=(20, 11))
    plt.boxplot(noise_list, labels=['Success', 'Failure'])
    plt.ylabel('SSIM')
    plt.legend(eps_list)

    plt.tight_layout()
    plt.savefig(fname)
    print("===== Finish =====")

def dot():
    fgsm = {
        'x': [3.91, 13.25, 30.42],
        'y': [0.2640, 0.3437, 0.3867],
        'std': [0.106, 0.101, 0.091]
    }
    ifgsm5 = {
        'x': [7.05, 25.05, 54.13],
        'y': [0.2385, 0.3187, 0.3617],
        'std': [0.119, 0.113, 0.105]
    }
    ifgsm40 = {
        'x': [8.15, 32.13, 70.69],
        'y': [0.2322, 0.3085, 0.3443],
        'std': [0.122, 0.117, 0.109]
    }
    gradfgsm = {
        'x': [56.67, 67.24, 74.18],
        'y': [0.3935, 0.4075, 0.4183],
        'std': [0.104, 0.096, 0.089]
    }
    plt.figure(figsize=(20, 11))
    plt.plot(fgsm['x'], fgsm['y'], label='FGSM', marker='o', markersize=14, linewidth=4)
    plt.plot(ifgsm5['x'], ifgsm5['y'], label='iFGSM (5)', marker='o', markersize=14, linewidth=4)
    plt.plot(ifgsm40['x'], ifgsm40['y'], label='iFGSM (40)', marker='o', markersize=14, linewidth=4)
    plt.plot(gradfgsm['x'], gradfgsm['y'], label='Grad-cam FGSM', marker='o', markersize=14, linewidth=4)
    plt.xticks(fontsize=16)
    plt.xlabel('Adversarial Success Rate', fontsize=24)
    plt.yticks(fontsize=16)
    plt.ylabel('MOD', fontsize=24)
    plt.legend(loc='lower right', fontsize=24)
    plt.tight_layout()
    plt.savefig('dot_plot.jpg')