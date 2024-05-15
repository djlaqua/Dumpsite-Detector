import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns


def calculate_iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def calculate_mean_iou(predicted_bbs, ground_truth_bbs):
    iou_scores = []
    for pred_box in predicted_bbs:
        iou_scores.append(max([calculate_iou(pred_box, gt_box) for gt_box in ground_truth_bbs]))
    return sum(iou_scores) / len(iou_scores) if iou_scores else 0

def is_full_image_box(box, image_shape):
    return np.array_equal(box, [0, 0, image_shape[1], image_shape[0]])


def custom_show(image, predicted_bbs, ground_truth_bbs, texts, metrics, is_true_negative):
    fig, ax = plt.subplots(1,figsize=(20, 8))
    ax.imshow(image)

    # Handle case where no dumpsites are detected
    if is_true_negative:
        plt.text(0.5, 0.5, 'Correctly detected no dumpsite', color='green', ha='center', va='center', transform=ax.transAxes, fontsize=12, backgroundcolor='black')
    elif len(predicted_bbs) == 0:
        plt.text(0.5, 0.5, 'No dumpsite detected', color='yellow', ha='center', va='center', transform=ax.transAxes, fontsize=12, backgroundcolor='black')

    # Add predicted bounding boxes (in red)
    for bb, text in zip(predicted_bbs, texts):
        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(bb[0], bb[1], text, color='white', backgroundcolor='red')

    # Add ground truth bounding boxes (in blue)
    for bb in ground_truth_bbs:
        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Display metrics on the image
    metrics_text = '\n'.join([f'{k}: {v:.2f}' for k, v in metrics.items()])
    plt.text(0.01, 0.01, metrics_text, color='yellow', transform=ax.transAxes, fontsize=10, backgroundcolor='black')

    plt.show()

def calculate_precision_recall(predicted_bbs, ground_truth_bbs, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0
    for pred_box in predicted_bbs:
        matched = False
        for gt_box in ground_truth_bbs:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matched = True
                break
        if matched:
            TP += 1
        else:
            FP += 1
    for gt_box in ground_truth_bbs:
        matched = False
        for pred_box in predicted_bbs:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matched = True
                break
        if not matched:
            FN += 1
    
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    return precision, recall
    
#    Evaluate the model on data from the loader and display results.
def evaluate_and_display(loader, model, decode_output, target2label, calculate_precision_recall, calculate_mean_iou, custom_show, max_batches=3):


    # Parameters:
    # loader: DataLoader for validation data.
    # model: The trained object detection model.
    # decode_output: Function to decode the model's output.
    # target2label: Mapping from target indices to labels.
    # calculate_precision_recall: Function to calculate precision and recall.
    # calculate_mean_iou: Function to calculate mean IoU.
    # custom_show: Function to display images, predictions, and metrics.
    # max_batches: Maximum number of batches to process.
   
    model.eval()
    for ix, (images, targets) in enumerate(loader):
        if ix >= max_batches: break
        images = [im for im in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            bbs, confs, labels = decode_output(output, target2label)
            gt_bbs = targets[i]['boxes'].cpu().numpy()

            # Check if the ground truth is a full-image box and no predictions were made
            full_image_gt = any(is_full_image_box(box, images[i].shape) for box in gt_bbs)
            predicted_empty = len(bbs) == 0

            if predicted_empty:
                precision, recall = 1.0, 1.0
                mIoU = np.nan  # mIoU is not meaningful in this context
            else:
                precision, recall = calculate_precision_recall(bbs, gt_bbs)
                mIoU = calculate_mean_iou(bbs, gt_bbs)

            metrics = {'mIoU': mIoU, 'Precision': precision, 'Recall': recall}
            image_to_show = images[i].cpu().permute(1, 2, 0).numpy()
            custom_show(image_to_show, bbs, gt_bbs, [f'{l}@{c:.2f}' for l, c in zip(labels, confs)], metrics, full_image_gt and predicted_empty)

def calculate_overall_metrics(loader, model, decode_output, target2label, iou_threshold=0.5):
    model.eval()
    total_TP, total_FP, total_FN = 0, 0, 0

    for images, targets in loader:
        images = [img for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            pred_bbs, _, _ = decode_output(output, target2label)
            gt_bbs = targets[i]['boxes'].cpu().numpy()

            matched_gt_boxes = set()
            for pred_box in pred_bbs:
                matched = False
                for j, gt_box in enumerate(gt_bbs):
                    if calculate_iou(pred_box, gt_box) >= iou_threshold and j not in matched_gt_boxes:
                        total_TP += 1
                        matched = True
                        matched_gt_boxes.add(j)
                        break
                if not matched:
                    total_FP += 1

            total_FN += len(gt_bbs) - len(matched_gt_boxes)

    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    accuracy = total_TP / (total_TP + total_FP + total_FN) if total_TP + total_FP + total_FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return accuracy, precision, recall, f1_score, total_TP, total_FP, total_FN



def plot_confusion_matrix(TP, FP, FN):
    confusion_matrix = np.array([[TP, FP], [FN, 0]])  # Using 0 for TN as it's not applicable
    labels = ['TP', 'FP', 'FN', 'TN']
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Positive', 'Predicted Negative'], yticklabels=['Actual Positive', 'Actual Negative'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.5, labels[i * 2 + j], ha='center', va='center', color='black')
    plt.show()
