import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path


# ---------------------------------------------------------------------
# AccuracyVisualizer for YOLOv3
# ---------------------------------------------------------------------
class AccuracyVisualizer:
    def __init__(self, save_dir='training_metrics'):
        self.save_dir = Path(save_dir)
        self.metrics_history = {
            'train/box_loss': [],
            'train/cls_loss': [],
            'val/box_loss': [],
            'val/cls_loss': [],
            'metrics/precision(B)': [],
            'metrics/recall(B)': [],
            'metrics/mAP50(B)': [],
            'metrics/mAP50-95(B)': []
        }

    def ensure_dir(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def update_metrics(self, results):
        if isinstance(results, dict):
            for k, v in results.items():
                if k in self.metrics_history:
                    self.metrics_history[k].append(v)

    def plot(self, epoch):
        self.ensure_dir()
        if not self.metrics_history['train/box_loss']:
            print("No data to plot yet.")
            return None

        epochs = range(1, len(self.metrics_history['train/box_loss']) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("YOLOv3 Training Metrics", fontsize=16, fontweight="bold")

        # 1️⃣ Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.metrics_history['train/box_loss'], 'b-', label='Train Box Loss')
        if self.metrics_history['val/box_loss']:
            ax1.plot(epochs, self.metrics_history['val/box_loss'], 'r-', label='Val Box Loss')
        ax1.set_title('Box Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2️⃣ Classification Loss
        ax2 = axes[0, 1]
        if self.metrics_history['train/cls_loss']:
            ax2.plot(epochs, self.metrics_history['train/cls_loss'], 'b-', label='Train Cls Loss')
        if self.metrics_history['val/cls_loss']:
            ax2.plot(epochs, self.metrics_history['val/cls_loss'], 'r-', label='Val Cls Loss')
        ax2.set_title('Cls Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3️⃣ Precision & Recall
        ax3 = axes[1, 0]
        if self.metrics_history['metrics/precision(B)']:
            ax3.plot(epochs, self.metrics_history['metrics/precision(B)'], 'g-', label='Precision')
        if self.metrics_history['metrics/recall(B)']:
            ax3.plot(epochs, self.metrics_history['metrics/recall(B)'], 'm-', label='Recall')
        ax3.legend()
        ax3.set_ylim(0, 1)
        ax3.set_title('Precision & Recall')
        ax3.grid(True, alpha=0.3)

        # 4️⃣ mAP
        ax4 = axes[1, 1]
        if self.metrics_history['metrics/mAP50(B)']:
            ax4.plot(epochs, self.metrics_history['metrics/mAP50(B)'], 'c-', label='mAP@50')
        if self.metrics_history['metrics/mAP50-95(B)']:
            ax4.plot(epochs, self.metrics_history['metrics/mAP50-95(B)'], 'y-', label='mAP@50-95')
        ax4.legend()
        ax4.set_ylim(0, 1)
        ax4.set_title('mAP Scores')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = self.save_dir / f'metrics_epoch_{epoch}.png'
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"📊 Metrics plot saved: {out_path}")


# ---------------------------------------------------------------------
# Train YOLOv3 model
# ---------------------------------------------------------------------
def train_yolov3(data_yaml, weights='yolov3.pt', epochs=10, imgsz=640, batch=16, device='0',
                 project='blood_cell_yolov3', name='exp'):
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    # Load data.yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    print(f"📁 Dataset: {data_yaml}")
    print(f"Classes ({data['nc']}): {data['names']}")

    visualizer = AccuracyVisualizer(save_dir=f'{project}/{name}/metrics')
    model = YOLO(weights)

    # Training arguments
    args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'project': project,
        'name': name,
        'device': device,
        'patience': 20,
        'exist_ok': True,
        'save': True,
        'optimizer': 'SGD',
    }

    print("🚀 Starting YOLOv3 Training...")
    results = model.train(**args)
    print("✅ Training Complete")

    # Post-training analysis
    results_csv = f"{project}/{name}/results.csv"
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        for _, row in df.iterrows():
            visualizer.update_metrics(row.to_dict())
        visualizer.plot(epochs)

        csv_out = f"{project}/{name}/training_metrics.csv"
        df.to_csv(csv_out, index=False)
        print(f"📈 Metrics CSV saved: {csv_out}")

    # Save training summary
    summary = {
        'best_model': f"{project}/{name}/weights/best.pt",
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'classes': data['names'],
        'nc': data['nc'],
    }
    summary_path = f"{project}/{name}/training_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f)
    print(f"📄 Training summary saved: {summary_path}")

    return results, summary['best_model']


# ---------------------------------------------------------------------
# Optional: Analyze results again separately
# ---------------------------------------------------------------------
def analyze_training_results(project_dir, experiment_name='exp'):
    csv_file = Path(project_dir) / experiment_name / 'results.csv'
    if not csv_file.exists():
        print(f"❌ No results.csv found at {csv_file}")
        return
    df = pd.read_csv(csv_file)
    epochs = range(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(epochs, df['metrics/mAP50(B)'], label='mAP@50')
    if 'metrics/precision(B)' in df.columns:
        ax.plot(epochs, df['metrics/precision(B)'], label='Precision')
    if 'metrics/recall(B)' in df.columns:
        ax.plot(epochs, df['metrics/recall(B)'], label='Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('YOLOv3 Accuracy Metrics')
    plt.tight_layout()
    out_path = csv_file.parent / 'training_analysis.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Analysis plot saved: {out_path}")


# ---------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv3 model with accuracy visualization")
    parser.add_argument('--data-yaml', type=str, default='blood_cell_yolo/data.yaml')
    parser.add_argument('--weights', type=str, default='yolov3.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--project', type=str, default='blood_cell_yolov3')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--analyze', action='store_true')

    args = parser.parse_args()

    if args.analyze:
        analyze_training_results(args.project, args.name)
    else:
        results, best = train_yolov3(
            data_yaml=args.data_yaml,
            weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name
        )
        analyze_training_results(args.project, args.name)
        print(f"\n🏁 Best model: {best}")
