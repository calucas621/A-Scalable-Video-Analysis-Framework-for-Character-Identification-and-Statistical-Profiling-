import matplotlib
matplotlib.use('Agg')  # 設置為 Agg 後端，避免使用 Tkinter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
import cv2
import numpy as np
import os
import pandas as pd
import threading
import time
import random as rd
from os.path import exists
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import ultralytics
from ultralytics import YOLO
import logging
import json
import pickle
from umap import UMAP
import albumentations as A
from fer import FER
from facenet_pytorch import InceptionResnetV1
from collections import defaultdict
import sys
import psutil
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import gc
import shutil
import torch
import tensorflow as tf
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#第一階段當前使用

# 將 DetectionModel 類加入白名單
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# 參數定義
MaxWaitSeconds = 8
MinWaitBetweenFrames = 5
MaxWaitBetweenFrames = 10
TimeOutSecs = 120

# 設置支援中文的字型（微軟正黑體）
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False
print(f"Matplotlib 後端: {matplotlib.get_backend()}")

# 檢查套件版本
print(f"PyTorch 版本: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}")
print(f"TensorFlow 版本: {tf.__version__}")
print(f"MediaPipe 版本: {mp.__version__}")
print(f"TensorFlow GPU 可用: {tf.config.list_physical_devices('GPU')}")

def setup_logging(output_dir, video_filename):
    """設置日誌系統，根據影片名稱生成唯一的日誌檔案"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{video_filename}_dbscan_cpu_new.log")
    try:
        test_file = os.path.join(log_dir, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("測試寫入")
        os.remove(test_file)
        print(f"目錄 {log_dir} 可寫入")
        handlers = [logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)
        logging.info("日誌系統初始化成功")
        logging.info(f"日誌將寫入: {log_file}")
    except Exception as e:
        print(f"日誌系統初始化失敗: {e}")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
        logging.warning("日誌檔案無法創建，已切換到控制台日誌")
    return log_file

class VideoAnalyzerStats:
    def __init__(self, output_dir, video_filename):
        """初始化統計分析模組"""
        self.output_dir = output_dir
        self.video_filename = video_filename
        self.cluster_timestamps = defaultdict(list)
        self.co_occurrence = defaultdict(lambda: defaultdict(int))
        self.cluster_main_colors = defaultdict(list)
        self.role_id_map = {}
        self._setup_directories()
        self._load_role_id_map()

    def _setup_directories(self):
        """設置輸出目錄"""
        self.stats_dir = os.path.join(self.output_dir, "stats", self.video_filename)
        os.makedirs(self.stats_dir, exist_ok=True)
        logging.info(f"統計輸出目錄已設置: {self.stats_dir}")

    def _load_role_id_map(self):
        """載入角色映射"""
        role_id_map_file = os.path.join(self.output_dir, f"role_id_map_{self.video_filename}.json")
        if os.path.exists(role_id_map_file):
            with open(role_id_map_file, 'r') as f:
                self.role_id_map = json.load(f)
            logging.info(f"已載入角色映射: {self.role_id_map}")
        else:
            logging.warning("角色映射檔案不存在，將使用 cluster_id 作為名稱")

    def update_data(self, cluster_id, timestamp, main_color, frame_clusters):
        """更新時間戳、服裝顏色和同框關係數據"""
        try:
            cluster_id_str = str(cluster_id)
            if cluster_id != -1:
                self.cluster_timestamps[cluster_id].append(timestamp)
                logging.debug(f"更新 cluster_timestamps，cluster_id: {cluster_id}, timestamp: {timestamp}")
            if cluster_id != -1 and main_color is not None and len(main_color) == 3:
                self.cluster_main_colors[cluster_id].append(main_color)
                logging.debug(f"更新 cluster_main_colors，cluster_id: {cluster_id}, main_color: {main_color}")
            for i, cid_i in enumerate(frame_clusters):
                for j, cid_j in enumerate(frame_clusters):
                    if i != j:
                        cid_i_str = str(cid_i) if cid_i != -1 else "unknown"
                        cid_j_str = str(cid_j) if cid_j != -1 else "unknown"
                        self.co_occurrence[cid_i_str][cid_j_str] += 1
                        logging.debug(f"更新 co_occurrence，{cid_i_str} -> {cid_j_str}: {self.co_occurrence[cid_i_str][cid_j_str]}")
        except Exception as e:
            logging.error(f"更新數據失敗: {e}")

    def _analyze_temporal_and_co_occurrence(self):
        """分析時間順序、同框關係和平均服裝顏色"""
        logging.info("開始分析時間順序和同框關係")
        temporal_stats = {}
        for cluster_id, timestamps in self.cluster_timestamps.items():
            if timestamps:
                timestamps = sorted(timestamps)
                cluster_name = self.role_id_map.get(str(cluster_id), f"角色_{cluster_id}")
                temporal_stats[cluster_name] = {
                    "timestamps": timestamps,
                    "count": len(timestamps),
                    "start_time": min(timestamps),
                    "end_time": max(timestamps),
                    "duration": max(timestamps) - min(timestamps)
                }
        logging.info(f"時間順序統計: {temporal_stats}")
        co_occurrence_stats = {}
        for cluster_id, relations in self.co_occurrence.items():
            if cluster_id == "unknown" and not relations:
                continue
            cluster_name = self.role_id_map.get(cluster_id, f"角色_{cluster_id}" if cluster_id != "unknown" else "未知角色")
            co_occurrence_stats[cluster_name] = {}
            for other_id, count in relations.items():
                if count > 0:
                    other_name = self.role_id_map.get(other_id, f"角色_{other_id}" if other_id != "unknown" else "未知角色")
                    co_occurrence_stats[cluster_name][other_name] = count
        logging.info(f"同框關係統計: {co_occurrence_stats}")
        color_stats = {}
        for cluster_id, colors in self.cluster_main_colors.items():
            if colors:
                cluster_name = self.role_id_map.get(str(cluster_id), f"角色_{cluster_id}")
                avg_color = np.mean(colors, axis=0).tolist()
                color_stats[cluster_name] = avg_color
        logging.info(f"平均服裝顏色: {color_stats}")
        self._save_stats_to_csv(temporal_stats, co_occurrence_stats, color_stats)
        self._visualize_stats(temporal_stats, co_occurrence_stats, color_stats)
        return temporal_stats, co_occurrence_stats, color_stats

    def _save_stats_to_csv(self, temporal_stats, co_occurrence_stats, color_stats):
        """將統計數據保存到 CSV"""
        try:
            temporal_data = []
            for name, stats in temporal_stats.items():
                temporal_data.append({
                    "CharacterName": name,
                    "AppearanceCount": stats["count"],
                    "StartTime": stats["start_time"],
                    "EndTime": stats["end_time"],
                    "Duration": stats["duration"]
                })
            temporal_df = pd.DataFrame(temporal_data)
            temporal_csv = os.path.join(self.stats_dir, "temporal_stats.csv")
            temporal_df.to_csv(temporal_csv, index=False)
            logging.info(f"時間順序統計已保存至: {temporal_csv}")
            co_occurrence_data = []
            for name, relations in co_occurrence_stats.items():
                for other_name, count in relations.items():
                    co_occurrence_data.append({
                        "Character1": name,
                        "Character2": other_name,
                        "CoOccurrenceCount": count
                    })
            co_occurrence_df = pd.DataFrame(co_occurrence_data)
            co_occurrence_csv = os.path.join(self.stats_dir, "co_occurrence_stats.csv")
            co_occurrence_df.to_csv(co_occurrence_csv, index=False)
            logging.info(f"同框關係統計已保存至: {co_occurrence_csv}")
            color_data = []
            for name, avg_color in color_stats.items():
                color_data.append({
                    "CharacterName": name,
                    "AvgColor_R": avg_color[0],
                    "AvgColor_G": avg_color[1],
                    "AvgColor_B": avg_color[2]
                })
            color_df = pd.DataFrame(color_data)
            color_csv = os.path.join(self.stats_dir, "color_stats.csv")
            color_df.to_csv(color_csv, index=False)
            logging.info(f"平均服裝顏色統計已保存至: {color_csv}")
        except Exception as e:
            logging.error(f"保存統計數據到 CSV 失敗: {e}")

    def _visualize_stats(self, temporal_stats, co_occurrence_stats, color_stats):
        """生成統計數據的可視化圖表"""
        try:
            plt.figure(figsize=(12, 6))
            for name, stats in temporal_stats.items():
                plt.hlines(y=name, xmin=stats["start_time"], xmax=stats["end_time"], linewidth=2)
            plt.xlabel("時間 (秒)")
            plt.ylabel("角色")
            plt.title("角色出現時間軸")
            plt.grid(True)
            temporal_plot = os.path.join(self.stats_dir, "temporal_timeline.png")
            plt.savefig(temporal_plot)
            plt.close()
            logging.info(f"時間軸圖已保存至: {temporal_plot}")
            if co_occurrence_stats:
                characters = sorted(set(co_occurrence_stats.keys()).union(*[set(r.keys()) for r in co_occurrence_stats.values()]))
                matrix = np.zeros((len(characters), len(characters)))
                for i, name1 in enumerate(characters):
                    for j, name2 in enumerate(characters):
                        matrix[i, j] = co_occurrence_stats.get(name1, {}).get(name2, 0)
                plt.figure(figsize=(10, 8))
                sns.heatmap(matrix, xticklabels=characters, yticklabels=characters, annot=True, fmt="g", cmap="Blues")
                plt.title("角色同框關係熱圖")
                co_occurrence_plot = os.path.join(self.stats_dir, "co_occurrence_heatmap.png")
                plt.savefig(co_occurrence_plot)
                plt.close()
                logging.info(f"同框關係熱圖已保存至: {co_occurrence_plot}")
            if color_stats:
                names = list(color_stats.keys())
                colors = np.array(list(color_stats.values()))
                plt.figure(figsize=(10, 6))
                for i, name in enumerate(names):
                    plt.bar(i, 1, color=colors[i], label=name)
                plt.xticks(range(len(names)), names, rotation=45)
                plt.title("角色平均服裝顏色")
                plt.legend()
                color_plot = os.path.join(self.stats_dir, "color_bars.png")
                plt.savefig(color_plot)
                plt.close()
                logging.info(f"服裝顏色圖已保存至: {color_plot}")
        except Exception as e:
            logging.error(f"生成可視化圖表失敗: {e}")

    def integrate_with_video_analyzer(self, video_analyzer):
        """與現有的 VideoAnalyzer 整合"""
        self.cluster_timestamps = video_analyzer.cluster_timestamps
        self.co_occurrence = video_analyzer.co_occurrence
        self.cluster_main_colors = video_analyzer.cluster_main_colors
        self.role_id_map = video_analyzer.role_id_map
        logging.info("已從 VideoAnalyzer 載入數據結構")
        return self._analyze_temporal_and_co_occurrence()

def check_dependencies():
    required_modules = [("cv2", "opencv-python"), ("numpy", "numpy"), ("pandas", "pandas"), ("mediapipe", "mediapipe"),
                        ("sklearn", "scikit-learn"), ("hdbscan", "hdbscan"), ("ultralytics", "ultralytics"),
                        ("umap", "umap-learn"), ("albumentations", "albumentations"), ("fer", "fer"),
                        ("torch", "torch"), ("facenet_pytorch", "facenet-pytorch"), ("seaborn", "seaborn"),
                        ("matplotlib", "matplotlib"), ("psutil", "psutil")]
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            logging.info(f"模組 {module_name} 已正確導入")
        except ImportError as e:
            logging.error(f"模組 {module_name} 導入失敗，請確保已安裝 {package_name}: {e}")
            print(f"錯誤：請安裝 {package_name} (pip install {package_name})")
            raise

def check_video_file(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"無法開啟影片檔案: {video_path}")
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or total_frames <= 0:
            logging.error(f"影片檔案無效: FPS={fps}, 總影格數={total_frames}")
            return False
        fps = int(fps)
        total_frames = int(total_frames)
        logging.info(f"影片資訊 - FPS: {fps}, 總影格數: {total_frames}")
        cap.release()
        return True
    except Exception as e:
        logging.error(f"檢查影片檔案失敗: {e}")
        return False

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    logging.info(f"當前記憶體使用量: {mem_mb:.2f} MB")
    return mem_mb

class CharacterFrame:
    def __init__(self, output_dir, video_filename):
        self.FrameID = 0
        self.FaceIdx = 0
        self.BodyIdx = 0
        self.FaceBox = []
        self.BodyBox = []
        self.FaceFile = ''
        self.ClusteredFaceFile = ''
        self.BodyFile = ''
        self.CharacterID = 0
        self.CharacterName = ''
        self.Expression = ''
        self.Confidence = 0.0
        self.FaceEmbedding = []
        self.MainColor = []
        self.csv_buffer = []
        self.csv_path = os.path.join(output_dir, "csv_files", f"character_features_{video_filename}.csv")
        output_dir_csv = os.path.dirname(self.csv_path)
        os.makedirs(output_dir_csv, exist_ok=True)
        test_file = os.path.join(output_dir_csv, "test.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("測試")
            os.remove(test_file)
            logging.info(f"目錄 {output_dir_csv} 可寫入")
        except PermissionError as e:
            logging.error(f"無權限寫入目錄 {output_dir_csv}: {e}")
            raise PermissionError(f"請確保有權限寫入 {output_dir_csv} 或更改輸出目錄")

    def save_to_csv(self):
        try:
            face_box_valid = len(self.FaceBox) >= 4
            body_box_valid = len(self.BodyBox) >= 4
            face_embedding = self.FaceEmbedding if self.FaceEmbedding and not np.any(np.isnan(self.FaceEmbedding)) else []
            main_color = self.MainColor if self.MainColor and not np.any(np.isnan(self.MainColor)) else []
            character_name = self.CharacterName if self.CharacterName else "未知角色"
            data = {
                "FrameID": self.FrameID,
                "FaceIdx": self.FaceIdx,
                "BodyIdx": self.BodyIdx,
                "FaceBoxX1": self.FaceBox[0] if face_box_valid else -1,
                "FaceBoxY1": self.FaceBox[1] if face_box_valid else -1,
                "FaceBoxX2": self.FaceBox[2] if face_box_valid else -1,
                "FaceBoxY2": self.FaceBox[3] if face_box_valid else -1,
                "BodyBoxX1": self.BodyBox[0] if body_box_valid else -1,
                "BodyBoxY1": self.BodyBox[1] if body_box_valid else -1,
                "BodyBoxX2": self.BodyBox[2] if body_box_valid else -1,
                "BodyBoxY2": self.BodyBox[3] if body_box_valid else -1,
                "FaceFile": self.FaceFile,
                "ClusteredFaceFile": self.ClusteredFaceFile,
                "BodyFile": self.BodyFile,
                "CharacterID": self.CharacterID,
                "CharacterName": character_name,
                "Expression": self.Expression,
                "Confidence": self.Confidence,
                "FaceEmbedding": json.dumps(face_embedding),
                "MainColor": json.dumps(main_color)
            }
            self.csv_buffer.append(data)
            logging.info(f"已添加數據至 CSV 緩衝區，當前緩衝區大小: {len(self.csv_buffer)}")
            if len(self.csv_buffer) >= 1:
                df = pd.DataFrame(self.csv_buffer)
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                try:
                    if exists(self.csv_path):
                        df.to_csv(self.csv_path, mode='a', header=False, index=False)
                    else:
                        df.to_csv(self.csv_path, mode='w', header=True, index=False)
                    logging.info(f"批次儲存 CSV 成功: {self.csv_path}, 寫入 {len(self.csv_buffer)} 筆數據")
                    self.csv_buffer = []
                except PermissionError as e:
                    logging.error(f"寫入 CSV 檔案 {self.csv_path} 時發生權限錯誤: {e}")
                    return False
                except Exception as e:
                    logging.error(f"寫入 CSV 檔案 {self.csv_path} 時發生錯誤: {e}")
                    return False
            return True
        except Exception as e:
            logging.error(f"save_to_csv 執行失敗: {e}")
            return False

    def flush_csv_buffer(self):
        if not self.csv_buffer:
            logging.info("CSV 緩衝區為空，無需寫入")
            return True
        try:
            df = pd.DataFrame(self.csv_buffer)
            if exists(self.csv_path):
                df.to_csv(self.csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.csv_path, mode='w', header=True, index=False)
            logging.info(f"剩餘緩衝區數據已儲存至 CSV: {self.csv_path}, 寫入 {len(self.csv_buffer)} 筆數據")
            self.csv_buffer = []
            return True
        except PermissionError as e:
            logging.error(f"強制寫入 CSV 檔案 {self.csv_path} 時發生權限錯誤: {e}")
            return False
        except Exception as e:
            logging.error(f"強制寫入 CSV 檔案 {self.csv_path} 時發生錯誤: {e}")
            return False

class VideoAnalyzer:
    def __init__(self, video_path, output_dir=r"C:\Users\user\Desktop\output", movie_title="Unknown",
                 subtitle_path=None, custom_checkpoint=None, reset_checkpoint=False, ascending=True,
                 stop_after_batch=False):
        try:
            logging.info("開始初始化 VideoAnalyzer")
            self.video_path = video_path
            self.video_filename = os.path.splitext(os.path.basename(video_path))[0]
            self.output_dir = os.path.join(output_dir, self.video_filename)
            self.subtitle_path = subtitle_path
            self.custom_checkpoint = custom_checkpoint
            self.ascending = ascending
            self.stop_after_batch = stop_after_batch
            self.frame_dir = os.path.join(self.output_dir, "frames")
            self.face_dir = os.path.join(self.output_dir, "faces")
            self.body_dir = os.path.join(self.output_dir, "bodies")
            self.csv_dir = os.path.join(self.output_dir, "csv_files")
            self.character_dir = os.path.join(self.output_dir, "character")
            self.BATCH_SIZE = 2000
            self.FLOW_THRESHOLD = 0.0
            self.CLUSTER_SIMILARITY_THRESHOLD = 0.85
            self.CLUSTER_DISTANCE_THRESHOLD = 5.0
            self.frames_processed = 0
            self.frames_per_batch = 500
            if not os.path.exists(self.video_path):
                logging.error(f"影片檔案不存在: {self.video_path}")
                raise FileNotFoundError(f"影片檔案不存在: {self.video_path}")
            self.device = torch.device("cpu")
            logging.info(f"使用設備: {self.device}")
            logging.info("初始化 MediaPipe 模型")
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)
            self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
            logging.info("初始化 YOLO 模型")
            yolo_model_path = r"C:\Users\user\Desktop\final video_detect\yolov8n.pt"
            if not os.path.exists(yolo_model_path):
                logging.error(f"YOLO 模型檔案不存在: {yolo_model_path}")
                raise FileNotFoundError(f"YOLO 模型檔案不存在: {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path).to(self.device)
            logging.info("初始化 FaceNet 模型")
            self.face_recognizer = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
            logging.info("初始化 FER 模型")
            self.emotion_detector = FER(mtcnn=True)
            haar_cascade_path = r"C:\Users\user\Desktop\final video_detect\haarcascade_frontalface_default.xml"
            if not os.path.exists(haar_cascade_path):
                logging.error(f"Haar Cascade 檔案不存在: {haar_cascade_path}")
                raise FileNotFoundError(f"Haar Cascade 檔案不存在: {haar_cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            if self.face_cascade.empty():
                logging.error(f"無法載入 Haar Cascade 模型，請檢查檔案是否正確: {haar_cascade_path}")
                raise FileNotFoundError(f"Haar Cascade 模型載入失敗: {haar_cascade_path}")
            else:
                logging.info(f"成功載入 Haar Cascade 模型: {haar_cascade_path}")
            self.feature_list = []
            self.pending_features = []
            self.pending_timestamps = []
            self.pending_main_colors = []
            self.feature_list_lock = threading.Lock()
            self.cluster_labels = None
            self.cluster_centers = {}
            self.role_id_map = {}
            self.prev_frame = None
            self.prev_faces = []
            self.no_face_count = 0
            self.cluster_timestamps = defaultdict(list)
            self.co_occurrence = defaultdict(lambda: defaultdict(int))
            self.cluster_main_colors = defaultdict(list)
            self.scaler = None
            self.pca = None
            self.pca_n_components = None
            self.scaler_pca_lock = threading.Lock()
            self.clustering_done = False
            self.total_features_extracted = 0
            self.batch_counter = 0
            self.last_cluster_time = 0
            self.checkpoint_file = os.path.join(self.output_dir, f"checkpoint_{self.video_filename}.txt")
            self.cluster_centers_file = os.path.join(self.output_dir, f"cluster_centers_{self.video_filename}.pkl")
            self._clear_old_files()
            if reset_checkpoint and os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                logging.info("檢查點檔案已重置")
            self.start_frame = self._load_checkpoint()
            self._setup_directories()
            self._load_role_id_map()
            self.augmentor = A.Compose([
                A.GaussNoise(p=0.3), A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.OneOf([A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2), A.RandomShadow(p=0.2)], p=0.3),
                A.Rotate(limit=15, p=0.5)
            ])
            self.stats_analyzer = VideoAnalyzerStats(self.output_dir, self.video_filename)
            logging.info("VideoAnalyzer 初始化成功")
        except Exception as e:
            logging.error(f"VideoAnalyzer 初始化失敗: {e}")
            raise

    def _clear_old_files(self):
        if os.path.exists(self.output_dir):
            for subdir in ["frames", "faces", "bodies", "character", "stats", "logs"]:
                subdir_path = os.path.join(self.output_dir, subdir)
                if os.path.exists(subdir_path):
                    for item in os.listdir(subdir_path):
                        item_path = os.path.join(subdir_path, item)
                        try:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                                logging.info(f"已刪除檔案: {item_path}")
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                                logging.info(f"已刪除目錄: {item_path}")
                        except PermissionError as e:
                            logging.warning(f"無法刪除 {item_path}，存取被拒: {e}")
                            try:
                                os.chmod(item_path, 0o666)
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                else:
                                    shutil.rmtree(item_path)
                                logging.info(f"更改權限後成功刪除: {item_path}")
                            except Exception as e2:
                                logging.error(f"更改權限後仍無法刪除 {item_path}: {e2}")
                                continue
                        except Exception as e:
                            logging.error(f"刪除 {item_path} 時發生錯誤: {e}")
                            continue
        csv_path = os.path.join(self.csv_dir, f"character_features_{self.video_filename}.csv")
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                logging.info("已清除舊 CSV 檔案")
            except PermissionError as e:
                logging.warning(f"無法刪除 CSV 檔案 {csv_path}，存取被拒: {e}")
            except Exception as e:
                logging.error(f"刪除 CSV 檔案 {csv_path} 時發生錯誤: {e}")

    def _setup_directories(self):
        os.makedirs(self.frame_dir, exist_ok=True)
        os.makedirs(self.face_dir, exist_ok=True)
        os.makedirs(self.body_dir, exist_ok=True)
        os.makedirs(self.character_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        logging.info("輸出目錄已設置完成")

    def _load_role_id_map(self):
        role_id_map_file = os.path.join(self.output_dir, f"role_id_map_{self.video_filename}.json")
        if os.path.exists(role_id_map_file):
            with open(role_id_map_file, 'r') as f:
                self.role_id_map = json.load(f)
            logging.info(f"已載入角色映射: {self.role_id_map}")
        else:
            self.role_id_map = {}
            logging.info("未找到角色映射檔案，將創建新映射")

    def _save_role_id_map(self):
        role_id_map_file = os.path.join(self.output_dir, f"role_id_map_{self.video_filename}.json")
        with open(role_id_map_file, 'w') as f:
            json.dump(self.role_id_map, f, indent=4)
        logging.info(f"角色映射已儲存至: {role_id_map_file}")

    def _load_checkpoint(self):
        if self.custom_checkpoint is not None:
            logging.info(f"使用自定義檢查點: {self.custom_checkpoint}")
            return self.custom_checkpoint
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = int(f.read().strip())
            logging.info(f"從檢查點檔案載入: {checkpoint}")
            return checkpoint
        logging.info("未找到檢查點檔案，從頭開始")
        return 0

    def _save_checkpoint(self, frame_id):
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(frame_id))
        logging.info(f"檢查點已儲存: {frame_id}")

    def _detect_faces(self, frame, frame_id):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detection.process(rgb_frame)
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append((x, y, width, height))
                    logging.info(f"影格 {frame_id} 檢測到人臉（MediaPipe），位置: ({x}, {y}), 尺寸: ({width}x{height}), 置信度: {detection.score[0]:.4f}")
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_haar = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                for (x, y, w, h) in faces_haar:
                    faces.append((x, y, w, h))
                    logging.info(f"影格 {frame_id} 檢測到人臉（Haar），位置: ({x}, {y}), 尺寸: ({w}x{h})")
            if faces:
                logging.info(f"影格 {frame_id} 最終檢測到 {len(faces)} 個人臉")
            return faces
        except Exception as e:
            logging.error(f"人臉檢測失敗: {e}")
            return []

    def _detect_body(self, frame, frame_id):
        try:
            results = self.yolo_model(frame)
            bodies = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.cls == 0:
                        x, y, w, h = map(int, box.xywh[0])
                        bodies.append((x - w // 2, y - h // 2, w, h))
            logging.info(f"影格 {frame_id} 檢測到 {len(bodies)} 個人體")
            return bodies
        except Exception as e:
            logging.error(f"人體檢測失敗: {e}")
            return []

    def _compute_optical_flow(self, frame, frame_id):
        try:
            if self.prev_frame is None or frame_id == 0:
                self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                logging.info(f"影格 {frame_id} 為初始影格，跳過光流計算")
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            self.prev_frame = gray
            return flow
        except Exception as e:
            logging.error(f"光流計算失敗: {e}")
            return None

    def _extract_face_embedding(self, face_img):
        if face_img is None or face_img.size == 0:
            logging.warning("人臉圖像為空，無法提取嵌入")
            return np.zeros(512)
        logging.info(f"提取人臉嵌入，圖像尺寸: {face_img.shape}")
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        augmented = self.augmentor(image=face_img_rgb)
        face_img_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        face_img_resized = cv2.resize(face_img_aug, (160, 160))
        face_img_tensor = torch.tensor(face_img_resized).permute(2, 0, 1).float() / 255.0
        face_img_tensor = face_img_tensor.unsqueeze(0).to(self.device)
        try:
            with torch.no_grad():
                embedding = self.face_recognizer(face_img_tensor).cpu().numpy().flatten()
            if len(embedding) != 512:
                logging.error(f"FaceNet 嵌入長度不正確: {len(embedding)}，應為 512")
                return np.zeros(512)
            log_memory_usage()
            return embedding
        except Exception as e:
            logging.error(f"FaceNet 特徵提取失敗: {e}")
            return np.zeros(512)

    def _extract_clothing_histogram(self, body_img):
        if body_img is None or body_img.size == 0:
            logging.warning("人體圖像為空，返回默認服裝顏色")
            return np.array([0.0, 0.0, 0.0])
        try:
            hsv = cv2.cvtColor(body_img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist = cv2.normalize(hist, hist).flatten()
            dominant_color = np.argmax(hist)
            return np.array([dominant_color / 180.0, 1.0, 1.0])
        except Exception as e:
            logging.error(f"服裝顏色提取失敗: {e}")
            return np.array([0.0, 0.0, 0.0])

    def _recognize_expression(self, face_img):
        try:
            result = self.emotion_detector.detect_emotions(face_img)
            if result:
                emotions = result[0]['emotions']
                return max(emotions, key=emotions.get)
            return "未知表情"
        except Exception as e:
            logging.error(f"表情識別失敗: {e}")
            return "未知表情"

    def _cosine_similarity(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def _histogram_similarity(self, h1, h2):
        h1 = np.array(h1, dtype='float32')
        h2 = np.array(h2, dtype='float32')
        return 1 - cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)

    def _merge_similar_clusters(self, all_clusters, reduced_features):
        try:
            logging.info("開始合併相似聚類...")
            cluster_ids = list(all_clusters.keys())
            n_clusters = len(cluster_ids)
            if n_clusters <= 50:
                logging.info(f"當前聚類數 {n_clusters} 已低於目標，無需合併")
                return all_clusters, self.cluster_labels
            similarity_matrix = np.zeros((n_clusters, n_clusters))
            for i, id_i in enumerate(cluster_ids):
                for j, id_j in enumerate(cluster_ids[i+1:], start=i+1):
                    if id_i == id_j:
                        continue
                    sim = self._cosine_similarity(
                        all_clusters[id_i]['avg_embedding'],
                        all_clusters[id_j]['avg_embedding']
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
            merge_map = {cid: cid for cid in cluster_ids}
            new_labels = self.cluster_labels.copy()
            merged_count = 0
            for i, id_i in enumerate(cluster_ids):
                for j, id_j in enumerate(cluster_ids[i+1:], start=i+1):
                    if similarity_matrix[i, j] > self.CLUSTER_SIMILARITY_THRESHOLD:
                        merge_map[id_j] = id_i
                        merged_count += 1
                        new_labels[self.cluster_labels == id_j] = id_i
            new_clusters = {}
            for old_id, new_id in merge_map.items():
                if new_id not in new_clusters:
                    new_clusters[new_id] = all_clusters[old_id].copy()
                    new_clusters[new_id]['timestamps'] = []
                    new_clusters[new_id]['main_colors'] = []
                    new_clusters[new_id]['embeddings'] = []
                new_clusters[new_id]['timestamps'].extend(all_clusters[old_id]['timestamps'])
                new_clusters[new_id]['main_colors'].extend(self.cluster_main_colors.get(old_id, []))
                new_clusters[new_id]['embeddings'].extend(
                    [self.feature_list[i][:512] for i in range(len(self.feature_list)) if self.cluster_labels[i] == old_id]
                )
            for new_id in new_clusters:
                if new_clusters[new_id]['embeddings']:
                    new_clusters[new_id]['avg_embedding'] = np.mean(new_clusters[new_id]['embeddings'], axis=0)
                if new_clusters[new_id]['main_colors']:
                    new_clusters[new_id]['main_color'] = np.mean(new_clusters[new_id]['main_colors'], axis=0)
                del new_clusters[new_id]['embeddings']
                del new_clusters[new_id]['main_colors']
            self.cluster_centers = {
                new_id: np.mean(reduced_features[new_labels == new_id], axis=0)
                for new_id in new_clusters if np.sum(new_labels == new_id) > 0
            }
            for old_id, new_id in merge_map.items():
                if old_id != new_id and str(old_id) in self.role_id_map:
                    self.role_id_map[str(new_id)] = self.role_id_map.get(str(new_id), self.role_id_map[str(old_id)])
                    del self.role_id_map[str(old_id)]
            new_timestamps = defaultdict(list)
            new_main_colors = defaultdict(list)
            for old_id, new_id in merge_map.items():
                new_timestamps[new_id].extend(self.cluster_timestamps.get(old_id, []))
                new_main_colors[new_id].extend(self.cluster_main_colors.get(old_id, []))
            self.cluster_timestamps = new_timestamps
            self.cluster_main_colors = new_main_colors
            new_co_occurrence = defaultdict(lambda: defaultdict(int))
            for old_id_i, counts in self.co_occurrence.items():
                new_id_i = merge_map.get(old_id_i, old_id_i)
                for old_id_j, count in counts.items():
                    new_id_j = merge_map.get(old_id_j, old_id_j)
                    new_co_occurrence[new_id_i][new_id_j] += count
            self.co_occurrence = new_co_occurrence
            n_new_clusters = len(set(new_labels) - {-1})
            logging.info(f"合併完成，合併 {merged_count} 個聚類，新聚類數: {n_new_clusters}")
            return new_clusters, new_labels
        except Exception as e:
            logging.error(f"合併相似聚類失敗: {e}")
            return all_clusters, self.cluster_labels

    def _perform_clustering(self, final=False):
        with self.feature_list_lock:
            if self.pending_features:
                self.feature_list.extend(self.pending_features)
                logging.info(f"已將 {len(self.pending_features)} 個待處理特徵加入 feature_list")
                pending_count = len(self.pending_features)
                self.pending_features = []
                pending_timestamps = self.pending_timestamps[:]
                pending_main_colors = self.pending_main_colors[:]
                self.pending_timestamps = []
                self.pending_main_colors = []
            else:
                pending_count = 0
                pending_timestamps = []
                pending_main_colors = []
            if len(self.feature_list) < 3:
                logging.warning(f"self.feature_list 長度 {len(self.feature_list)} 小於 3，跳過聚類")
                return False
            try:
                log_memory_usage()
                logging.info("開始轉換特徵矩陣...")
                features_matrix = np.array(self.feature_list, dtype=np.float32)
                logging.info(f"特徵矩陣創建完成，形狀: {features_matrix.shape}")
                if np.any(np.isnan(features_matrix)) or np.any(np.isinf(features_matrix)):
                    logging.warning("特徵矩陣包含無效值，清理數據")
                    features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception as e:
                logging.error(f"轉換特徵矩陣失敗: {e}", exc_info=True)
                return False

        with self.scaler_pca_lock:
            try:
                log_memory_usage()
                logging.info("開始特徵縮放...")
                self.scaler = StandardScaler()
                features_scaled = self.scaler.fit_transform(features_matrix)
                logging.info(f"特徵縮放完成，形狀: {features_scaled.shape}")
                log_memory_usage()
                logging.info("開始 PCA 降維...")
                n_samples, n_features = features_matrix.shape
                n_components = min(3, min(n_samples, n_features))
                self.pca = PCA(n_components=n_components)
                self.pca_n_components = n_components
                reduced_features = self.pca.fit_transform(features_scaled)
                logging.info(f"PCA 降維完成，n_components={n_components}, 形狀: {reduced_features.shape}")
                scaler_file = os.path.join(self.output_dir, f"scaler_{self.video_filename}.pkl")
                pca_file = os.path.join(self.output_dir, f"pca_{self.video_filename}.pkl")
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scaler, f)
                with open(pca_file, 'wb') as f:
                    pickle.dump(self.pca, f)
                logging.info(f"Scaler 和 PCA 已儲存至: {scaler_file}, {pca_file}")
                log_memory_usage()
                logging.info("開始 HDBSCAN 聚類...")
                clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, cluster_selection_epsilon=0.5)
                self.cluster_labels = clusterer.fit_predict(reduced_features)
                n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
                noise_points = np.sum(self.cluster_labels == -1)
                logging.info(f"聚類完成，聚類數: {n_clusters}，噪聲點數量: {noise_points}")
                logging.info(f"當前 cluster_labels: {self.cluster_labels[:10]}...")
                log_memory_usage()
                logging.info("計算聚類中心...")
                unique_labels = set(self.cluster_labels)
                existing_centers = {}
                if os.path.exists(self.cluster_centers_file):
                    with open(self.cluster_centers_file, 'rb') as f:
                        existing_centers = pickle.load(f)
                self.cluster_centers = existing_centers.copy()
                for label in unique_labels:
                    if label != -1:
                        points = reduced_features[self.cluster_labels == label]
                        center = np.mean(points, axis=0)
                        if str(label) not in self.cluster_centers:
                            self.cluster_centers[label] = center
                with open(self.cluster_centers_file, 'wb') as f:
                    pickle.dump(self.cluster_centers, f)
                logging.info(f"聚類中心已儲存至: {self.cluster_centers_file}")
                logging.info("開始分配角色名稱...")
                all_clusters = {}
                for label in unique_labels:
                    if label == -1:
                        continue
                    label_str = str(label)
                    if label_str in self.role_id_map:
                        continue
                    main_colors = self.cluster_main_colors.get(label, [])
                    timestamps = self.cluster_timestamps.get(label, [])
                    embeddings = [self.feature_list[i][:512] for i in range(len(self.feature_list)) if self.cluster_labels[i] == label]
                    cluster_features = {
                        'avg_embedding': np.mean(embeddings, axis=0) if embeddings else np.zeros(512),
                        'main_color': np.mean(main_colors, axis=0) if main_colors else np.zeros(3),
                        'timestamps': timestamps
                    }
                    all_clusters[label] = cluster_features
                    self.role_id_map[label_str] = f"角色_{label}"
                    logging.info(f"Cluster {label} 分配角色名稱: 角色_{label}")
                all_clusters, self.cluster_labels = self._merge_similar_clusters(all_clusters, reduced_features)
                self._save_role_id_map()
                feature_offset = len(self.feature_list) - pending_count
                for i, label in enumerate(self.cluster_labels[feature_offset:]):
                    if label != -1 and i < len(pending_timestamps):
                        self.cluster_timestamps[label].append(pending_timestamps[i])
                        self.cluster_main_colors[label].append(pending_main_colors[i])
                        logging.debug(f"分配 pending 數據: label={label}, timestamp={pending_timestamps[i]}, main_color={pending_main_colors[i]}")
                self.clustering_done = True
                logging.info("聚類成功完成，設置 clustering_done 為 True")
                if not final:
                    output_file = f"cluster_visualization_batch_{self.batch_counter}_{self.video_filename}.png"
                    self._visualize_clusters(reduced_features, output_file)
                    self.batch_counter += 1
                if final:
                    output_file = f"cluster_visualization_final_{self.video_filename}.png"
                    self._visualize_clusters(reduced_features, output_file)
                self.feature_list = []
                gc.collect()
                log_memory_usage()
                return True
            except Exception as e:
                logging.error(f"_perform_clustering 執行失敗: {e}", exc_info=True)
                self.clustering_done = False
                return False

    def _predict_cluster(self, feature_vector):
        try:
            if not self.clustering_done:
                logging.warning("預測聚類失敗：clustering_done 為 False")
                return -1
            if not self.cluster_centers:
                logging.warning("預測聚類失敗：cluster_centers 為空")
                return -1
            if self.scaler is None or self.pca is None:
                logging.warning("預測聚類失敗：Scaler 或 PCA 未初始化")
                return -1
            with self.scaler_pca_lock:
                feature_scaled = self.scaler.transform([feature_vector])
                feature_reduced = self.pca.transform(feature_scaled)
            distances = {label: np.linalg.norm(feature_reduced[0] - center) for label, center in self.cluster_centers.items()}
            nearest_cluster = min(distances, key=distances.get)
            if distances[nearest_cluster] > self.CLUSTER_DISTANCE_THRESHOLD:
                logging.info(f"特徵向量與最近聚類 {nearest_cluster} 的距離 {distances[nearest_cluster]:.2f} 過大，返回 -1")
                return -1
            logging.info(f"特徵向量分配到聚類 {nearest_cluster}，距離: {distances[nearest_cluster]:.2f}")
            return nearest_cluster
        except Exception as e:
            logging.error(f"預測聚類失敗: {e}")
            return -1

    def _visualize_clusters(self, reduced_features, output_file):
        try:
            if reduced_features.shape[1] < 2:
                logging.warning(f"降維特徵維度 {reduced_features.shape[1]} 小於 2，無法可視化")
                return
            logging.info("開始 UMAP 降維以進行可視化...")
            umap_reducer = UMAP(n_components=2, random_state=42)
            umap_features = umap_reducer.fit_transform(reduced_features)
            mask = self.cluster_labels != -1
            filtered_umap_features = umap_features[mask]
            filtered_labels = self.cluster_labels[mask]
            if len(filtered_umap_features) == 0:
                logging.warning("過濾後無有效數據點，無法可視化")
                return
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(filtered_umap_features[:, 0], filtered_umap_features[:, 1], c=filtered_labels, cmap='viridis', s=50)
            plt.colorbar(scatter, label='聚類標籤')
            plt.xlabel('UMAP 維度 1')
            plt.ylabel('UMAP 維度 2')
            plt.title('角色聚類結果（UMAP 可視化，排除噪聲點）')
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path)
            plt.close()
            logging.info(f"聚類可視化已儲存至: {output_path}")
        except Exception as e:
            logging.error(f"聚類可視化失敗: {e}")

    def _analyze_temporal_and_co_occurrence(self):
        logging.info("開始整合統計數據並進行分析...")
        self.stats_analyzer.cluster_timestamps = self.cluster_timestamps
        self.stats_analyzer.co_occurrence = self.co_occurrence
        self.stats_analyzer.cluster_main_colors = self.cluster_main_colors
        self.stats_analyzer.role_id_map = self.role_id_map
        self.stats_analyzer._analyze_temporal_and_co_occurrence()

    def process_frame(self, frame_id):
        try:
            start_time = time.time()
            mem_mb = log_memory_usage()
            if mem_mb > 5000:
                logging.warning("記憶體使用量過高，清理緩存並暫停...")
                gc.collect()
                time.sleep(5)
            zero_based_frame_id = frame_id - self.start_frame
            logging.info(f"處理影格 FrameID: {frame_id} (零基索引: {zero_based_frame_id}) 開始")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logging.error(f"無法開啟影片: {self.video_path}")
                return True
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_seconds = total_frames // fps
            if frame_id >= total_seconds * 2:
                logging.warning(f"影格 {frame_id} 超出影片範圍（總秒數: {total_seconds}），跳過")
                cap.release()
                return True
            actual_seconds = frame_id / 2
            frame_position = int(actual_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            timestamp = actual_seconds
            cap.release()
            if not ret:
                logging.error(f"無法讀取影格 {frame_id}，可能是影片損壞或索引錯誤")
                return True
            logging.info(f"成功讀取影格 {frame_id}，尺寸: {frame.shape}")
            frame = cv2.resize(frame, (1280, 720))
            frame_filename = os.path.join(self.frame_dir, f"frame_{zero_based_frame_id}.jpg")
            flow = self._compute_optical_flow(frame, frame_id)
            if flow is not None:
                mean_flow = np.mean(np.abs(flow))
                if mean_flow < self.FLOW_THRESHOLD:
                    logging.info(f"跳過影格 {frame_id}，運動量過低: {mean_flow}")
                    return True
            faces = self._detect_faces(frame, frame_id)
            bodies = self._detect_body(frame, frame_id)
            logging.info(f"影格 {frame_id} 檢測到 {len(faces)} 個人臉，{len(bodies)} 個人體")
            frame_data = []
            frame_clusters = []
            if not faces:
                self.no_face_count += 1
                logging.warning(f"影格 {frame_id} 未檢測到人臉，連續無人臉影格數: {self.no_face_count}")
                if self.no_face_count <= 10 and self.prev_faces:
                    logging.info(f"使用前一影格的人臉位置進行補償，prev_faces: {self.prev_faces}")
                    faces = self.prev_faces
                else:
                    self.prev_faces = []
            else:
                self.no_face_count = 0
                self.prev_faces = faces
            if not faces:
                logging.warning(f"影格 {frame_id} 最終仍未檢測到人臉，跳過特徵提取")
                self.frames_processed += 1
                return True
            current_time = time.time()
            for face_idx, (fx, fy, fw, fh) in enumerate(faces):
                if fx < 0 or fy < 0 or fw <= 0 or fh <= 0 or fx + fw > frame.shape[1] or fy + fh > frame.shape[0]:
                    logging.warning(f"無效的人臉框，跳過: ({fx}, {fy}, {fw}, {fh})")
                    continue
                face_img = frame[fy:fy+fh, fx:fx+fw]
                if face_img.size == 0:
                    logging.warning(f"人臉圖像為空，跳過: face_idx={face_idx}")
                    continue
                body_idx = -1
                min_dist = float('inf')
                for idx, (bx, by, bw, bh) in enumerate(bodies):
                    dist = np.sqrt((fx - bx)**2 + (fy - by)**2)
                    if dist < min_dist:
                        min_dist = dist
                        body_idx = idx
                body_box = bodies[body_idx] if body_idx != -1 else None
                body_img = frame[body_box[1]:body_box[1]+body_box[3], body_box[0]:body_box[0]+body_box[2]] if body_box else None
                face_embedding = self._extract_face_embedding(face_img)
                main_color = self._extract_clothing_histogram(body_img)
                feature_vector = np.concatenate([face_embedding, main_color])
                if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                    logging.warning(f"影格 {frame_id} 特徵向量包含無效值，使用零向量替代")
                    feature_vector = np.zeros(515)
                self.total_features_extracted += 1
                logging.info(f"總計提取特徵數: {self.total_features_extracted}")
                with self.feature_list_lock:
                    self.feature_list.append(feature_vector)
                    logging.info(f"已添加特徵向量，feature_list 長度: {len(self.feature_list)}, 特徵內容: {feature_vector[:10]}...")
                    if not self.clustering_done and len(self.feature_list) >= self.BATCH_SIZE:
                        logging.info("首次聚類未完成，強制執行聚類")
                        success = self._perform_clustering(final=False)
                        if not success:
                            logging.error("首次聚類失敗，將特徵移至 pending_features")
                            self.pending_features.extend(self.feature_list)
                            self.feature_list = []
                    character_id = self._predict_cluster(feature_vector)
                    frame_clusters.append(character_id)
                    logging.debug(f"影格 {frame_id} frame_clusters: {frame_clusters}")
                    self.stats_analyzer.update_data(character_id, timestamp, main_color, frame_clusters)
                    if character_id != -1:
                        self.cluster_timestamps[character_id].append(timestamp)
                        self.cluster_main_colors[character_id].append(main_color)
                        logging.info(f"影格 {frame_id} 分配到聚類 {character_id}")
                    else:
                        self.pending_features.append(feature_vector)
                        self.pending_timestamps.append(timestamp)
                        self.pending_main_colors.append(main_color)
                        logging.warning(f"影格 {frame_id} 預測聚類為 -1，儲存至 pending_features，當前待處理特徵數: {len(self.pending_features)}")
            self.frames_processed += 1
            if self.frames_processed % self.frames_per_batch == 0 and len(self.feature_list) >= 3:
                logging.info(f"已處理 {self.frames_processed} 個影格，執行 batch 聚類")
                success = self._perform_clustering(final=False)
                if success:
                    self.feature_list = []
                    logging.info("特徵列表已清空")
                    self.last_cluster_time = current_time
                    self._analyze_temporal_and_co_occurrence()
                else:
                    logging.error("聚類執行失敗")
            if flow is not None and len(frame_data) > 0 and mean_flow < self.FLOW_THRESHOLD:
                prev_cf = frame_data[-1]
                if abs(prev_cf.FrameID - frame_id) == 1:
                    character_id = prev_cf.CharacterID
            character_name = self.role_id_map.get(str(character_id), f"角色_{character_id}" if character_id != -1 else "未知角色")
            char_info = {"name": character_name, "output_path": os.path.join(self.character_dir, character_name if character_id != -1 else "unknown")}
            os.makedirs(char_info["output_path"], exist_ok=True)
            cf = CharacterFrame(self.output_dir, self.video_filename)
            cf.FrameID = frame_id
            cf.FaceIdx = face_idx
            cf.FaceBox = [fx, fy, fx+fw, fy+fh]
            cf.Expression = self._recognize_expression(face_img)
            cf.Confidence = 0.8 if character_id != -1 else 0.0
            cf.CharacterID = character_id
            cf.CharacterName = char_info["name"]
            cf.FaceEmbedding = face_embedding.tolist()
            cf.MainColor = main_color.tolist()
            if body_img is not None and body_img.size > 0:
                cf.BodyIdx = body_idx
                cf.BodyBox = body_box
                cf.BodyFile = os.path.join(self.body_dir, f"body_{zero_based_frame_id}_{body_idx}.jpg")
                cv2.imwrite(cf.BodyFile, body_img)
            cf.FaceFile = os.path.join(self.face_dir, f"face_{zero_based_frame_id}_{face_idx}.jpg")
            cv2.imwrite(cf.FaceFile, face_img)
            cf.ClusteredFaceFile = os.path.join(char_info["output_path"], f"face_{zero_based_frame_id}_{cf.FaceIdx}.jpg")
            cv2.imwrite(cf.ClusteredFaceFile, face_img)
            frame_data.append(cf)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            cv2.putText(frame, f"{cf.CharacterName} (ID: {cf.CharacterID})", (fx, fy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if cf.BodyBox:
                cv2.rectangle(frame, (cf.BodyBox[0], cf.BodyBox[1]), (cf.BodyBox[0] + cf.BodyBox[2], cf.BodyBox[1] + cf.BodyBox[3]), (255, 0, 0), 2)
            if frame_id % 10 == 0:
                cv2.imwrite(frame_filename, frame)
            for cf in frame_data:
                cf.save_to_csv()
            self._save_checkpoint(frame_id)
            logging.info(f"影格 {frame_id} 處理完成，耗時: {time.time() - start_time:.2f} 秒")
            if time.time() - start_time > 15:
                logging.warning(f"影格 {frame_id} 處理超時，強制結束")
                return True
            return True
        except Exception as e:
            logging.error(f"[FRAME ERROR] 影格 {frame_id} 執行中發生錯誤，跳過處理: {e}")
            print(f"!!! 影格 {frame_id} 停止處理（回傳 True 以繼續處理後續影格）")
            return True

    def analyze(self):
        self.stop_after_batch = False
        if not os.path.exists(self.video_path):
            logging.error(f"影片檔案不存在: {self.video_path}")
            return
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or total_frames <= 0:
            logging.error(f"影片無效: FPS={fps}, 總影格數={total_frames}")
            return
        start_frame = max(self.start_frame, 0)  # 確保 start_frame 不為負
        total_seconds = total_frames // fps
        logging.info(f"影片總秒數: {total_seconds}, 總影格數: {total_frames}, 起始影格: {start_frame}")
        if self.ascending:
            frame_range = range(start_frame * 2, total_seconds * 2, 2)
        else:
            frame_range = range(total_seconds * 2 - 1, start_frame * 2 - 1, -2)
        logging.info(f"開始處理影格，共 {len(frame_range)} 個")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_frame = {}
            frame_iter = iter(frame_range)
            for _ in range(1):
                try:
                    frame_id = next(frame_iter)
                    future = executor.submit(self.process_frame, frame_id)
                    future_to_frame[future] = frame_id
                except StopIteration:
                    break
            while future_to_frame:
                done, _ = wait(future_to_frame, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    frame_id = future_to_frame.pop(future)
                    try:
                        result = future.result()
                        logging.info(f"影格 {frame_id} 處理結果: {result}")
                    except Exception as e:
                        logging.error(f"影格 {frame_id} 發生例外錯誤: {e}", exc_info=True)
                    try:
                        next_frame_id = next(frame_iter)
                        new_future = executor.submit(self.process_frame, next_frame_id)
                        future_to_frame[new_future] = next_frame_id
                    except StopIteration:
                        continue
        cf = CharacterFrame(self.output_dir, self.video_filename)
        cf.flush_csv_buffer()
        if len(self.feature_list) > 0 or len(self.pending_features) > 0:
            logging.info(f"最終聚類，feature_list 長度: {len(self.feature_list)}, pending_features 長度: {len(self.pending_features)}")
            self._perform_clustering(final=True)
            self._analyze_temporal_and_co_occurrence()
            self.feature_list = []
            self.pending_features = []
            self.pending_timestamps = []
            self.pending_main_colors = []
        logging.info("影片處理完成，已生成最終聚類圖")

def match_character_names(output_dir, video_filename, character_csv_path, role_id_map_path):
    if not os.path.exists(character_csv_path):
        logging.error(f"CSV 檔案不存在: {character_csv_path}")
        return
    try:
        df = pd.read_csv(character_csv_path)
        logging.info(f"從 {character_csv_path} 載入 {len(df)} 筆數據")
        if not os.path.exists(role_id_map_path):
            logging.warning("角色映射檔案不存在，跳過角色名稱匹配")
            return
        with open(role_id_map_path, 'r') as f:
            role_id_map = json.load(f)
        df['CharacterName'] = df['CharacterID'].astype(str).map(role_id_map).fillna(df['CharacterName'])
        df.to_csv(character_csv_path, index=False)
        logging.info(f"角色名稱已更新並儲存至: {character_csv_path}")
    except Exception as e:
        logging.error(f"匹配角色名稱失敗: {e}")

def evaluate_accuracy(output_dir, video_filename, character_csv_path, ground_truth_csv=None, ground_truth_corrected=None):
    if not os.path.exists(character_csv_path):
        logging.error(f"CSV 檔案不存在: {character_csv_path}")
        return
    try:
        df = pd.read_csv(character_csv_path)
        total_frames = len(df)
        unknown_frames = len(df[df['CharacterName'] == "未知角色"])
        accuracy = 1 - (unknown_frames / total_frames) if total_frames > 0 else 0
        logging.info(f"識別準確率: {accuracy:.2%} (總影格: {total_frames}, 未識別影格: {unknown_frames})")
        accuracy_file = os.path.join(output_dir, f"accuracy_{video_filename}.txt")
        with open(accuracy_file, 'w') as f:
            f.write(f"識別準確率: {accuracy:.2%}\n")
            f.write(f"總影格數: {total_frames}\n")
            f.write(f"未識別影格數: {unknown_frames}\n")
        logging.info(f"準確率報告已生成: {accuracy_file}")
        if ground_truth_csv and ground_truth_corrected and os.path.exists(ground_truth_csv):
            try:
                gt_df = pd.read_csv(ground_truth_csv)
                merged_df = df.merge(gt_df, on="FrameID", suffixes=('_pred', '_true'))
                correct_matches = (merged_df['CharacterName_pred'] == merged_df['TrueCharacterName']).sum()
                total_matches = len(merged_df)
                gt_accuracy = correct_matches / total_matches if total_matches > 0 else 0
                logging.info(f"基於 ground truth 的準確率: {gt_accuracy:.2%} (正確匹配: {correct_matches}, 總匹配: {total_matches})")
                with open(ground_truth_corrected, 'w') as f:
                    f.write(f"基於 ground truth 的準確率: {gt_accuracy:.2%}\n")
                    f.write(f"正確匹配數: {correct_matches}\n")
                    f.write(f"總匹配數: {total_matches}\n")
                logging.info(f"Ground truth 準確率報告已儲存至: {ground_truth_corrected}")
            except Exception as e:
                logging.error(f"基於 ground truth 評估準確率失敗: {e}")
    except Exception as e:
        logging.error(f"評估準確率失敗: {e}")

def main():
    try:
        video_path = r"C:\Users\user\Desktop\output\sample\63_鈴芽之旅.mp4"
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        character_csv_path = os.path.join(r"C:\Users\user\Desktop\output", "csv_files", f"character_features_{video_filename}.csv")
        role_id_map_path = os.path.join(r"C:\Users\user\Desktop\output", f"role_id_map_{video_filename}.json")
        output_dir = r"C:\Users\user\Desktop\output"
        ground_truth_csv = r"C:\Users\user\Desktop\output\ground_truth.csv"
        ground_truth_corrected = r"C:\Users\user\Desktop\output\ground_truth_corrected.csv"
        movie_title = "Unknown"
        subtitle_path = None
        custom_checkpoint = None
        reset_checkpoint = False
        ascending = True
        stop_after_batch = False

        logging.info("程式開始執行")
        check_dependencies()
        if not check_video_file(video_path):
            raise FileNotFoundError("影片檔案無法開啟")
        setup_logging(output_dir, video_filename)
        analyzer = VideoAnalyzer(
            video_path=video_path,
            output_dir=output_dir,
            movie_title=movie_title,
            subtitle_path=subtitle_path,
            custom_checkpoint=custom_checkpoint,
            reset_checkpoint=reset_checkpoint,
            ascending=ascending,
            stop_after_batch=stop_after_batch
        )
        analyzer.analyze()
        match_character_names(output_dir, video_filename, character_csv_path, role_id_map_path)
        evaluate_accuracy(output_dir, video_filename, character_csv_path, ground_truth_csv, ground_truth_corrected)
        logging.info("程式執行完畢")
    except Exception as e:
        logging.error(f"程式執行過程中發生錯誤: {e}", exc_info=True)
        print(f"程式執行過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()