import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTModel
from tqdm import tqdm
import logging
import json
import os
import psutil
from scipy.signal import find_peaks
import multiprocessing

#第二階段 場景分割(目前 最新)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(r"C:\Users\user\Desktop\output\scene_segmenter_ml.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class SceneDataset(Dataset):
    def __init__(self, video_path, frame_to_chars, ground_truth_csv, fps=30, k=60, stride=20, segment_idx=0, samples_per_segment=1235):
        self.video_path = video_path
        self.frame_to_chars = frame_to_chars
        self.ground_truth = pd.read_csv(ground_truth_csv)
        self.fps = fps
        self.k = k
        self.stride = stride
        self.segment_idx = segment_idx
        self.samples_per_segment = samples_per_segment
        self.frame_cache = {}
        self.flow_cache = self._precompute_flow()
        self.strong_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
        self.weak_aug = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.samples = self._create_samples()
        self.unique_chars = sorted(set().union(*frame_to_chars.values()) - {'未知角色'})
        self.role_dim = max(len(self.unique_chars), 1)
        logging.info(f"初始化 SceneDataset，分段 {segment_idx}，唯一角色數: {self.role_dim}")

    def _create_samples(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logging.error(f"無法打開影片: {self.video_path}")
                return []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = self.segment_idx * self.samples_per_segment * self.stride
            end_frame = min(start_frame + self.samples_per_segment * self.stride, total_frames - self.k + 1)
            samples = []
            for frame_id in tqdm(range(start_frame, end_frame, self.stride), desc=f"生成樣本（分段 {self.segment_idx}）"):
                timestamp = frame_id / self.fps
                end_frame_id = frame_id + self.k
                end_timestamp = end_frame_id / self.fps
                cond1 = (self.ground_truth['start_time_seconds'].between(timestamp, end_timestamp)).any()
                cond2 = (self.ground_truth['end_time_seconds'].between(timestamp, end_timestamp)).any()
                label = 1 if cond1 or cond2 else 0
                samples.append((frame_id, end_frame_id, label, timestamp, end_timestamp))
            cap.release()
            logging.info(f"分段 {self.segment_idx} 生成樣本數: {len(samples)}")
            return samples
        except Exception as e:
            logging.error(f"生成樣本失敗（分段 {self.segment_idx}）: {e}")
            return []

    def _precompute_flow(self):
        try:
            flow_cache_path = os.path.join(r"C:\Users\user\Desktop\output", f"flow_cache_segment_{self.segment_idx}.npy")
            if os.path.exists(flow_cache_path):
                logging.info(f"載入光流快取: {flow_cache_path}")
                return np.load(flow_cache_path, allow_pickle=True).item()
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logging.error(f"無法打開影片: {self.video_path}")
                return {}
            cv2.setNumThreads(multiprocessing.cpu_count())  # 設置 OpenCV 線程數
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            flow_cache = {}
            segment_size = 20
            num_segments = 2  # 固定為 2，減少計算
            for frame_id, end_frame_id, _, timestamp, end_timestamp in tqdm(self.samples, desc=f"預計算光流（分段 {self.segment_idx}）"):
                flow_key = (timestamp, end_timestamp)
                if flow_key not in flow_cache:
                    start_frame = int(timestamp * self.fps)
                    end_frame = min(int(end_timestamp * self.fps), total_frames)
                    flow_values = []
                    for i in range(num_segments):
                        segment_start = start_frame + i * segment_size
                        segment_end = min(segment_start + segment_size, end_frame)
                        segment_flow = []
                        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start)
                        ret, prev_frame = cap.read()
                        if not ret:
                            break
                        for f in range(segment_start + 1, segment_end):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            flow_mag = np.mean(np.sqrt(np.sum(flow ** 2, axis=2)))
                            segment_flow.append(flow_mag)
                            prev_frame = frame
                        if segment_flow:
                            flow_values.append(np.mean(segment_flow))
                    flow_cache[flow_key] = sum(flow_values) * (segment_size / self.fps) if flow_values else 0.0
            cap.release()
            np.save(flow_cache_path, flow_cache)
            logging.info(f"光流預計算完成，分段 {self.segment_idx}")
            return flow_cache
        except Exception as e:
            logging.error(f"預計算光流失敗（分段 {self.segment_idx}）: {e}")
            return {}

    def _load_frame_chunk(self, frame_id):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logging.error(f"無法打開影片: {self.video_path}")
                return
            start_frame = max(0, frame_id - 20)
            for i in range(start_frame, frame_id + self.k, self.stride):
                if i in self.frame_cache:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
                    self.frame_cache[i] = frame
                else:
                    self.frame_cache[i] = np.zeros((224, 224, 3), dtype=np.float32)
            cap.release()
            if len(self.frame_cache) > 30:
                self.frame_cache = {k: v for k, v in list(self.frame_cache.items())[-30:]}
        except Exception as e:
            logging.error(f"載入影格塊 {frame_id} 失敗: {e}")

    def _encode_roles(self, chars):
        role_vec = torch.zeros(self.role_dim, dtype=torch.float32)
        for char in chars:
            if char in self.unique_chars:
                role_vec[self.unique_chars.index(char)] = 1.0
        return role_vec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            frame_id, end_frame_id, label, timestamp, end_timestamp = self.samples[idx]
            if frame_id not in self.frame_cache:
                self._load_frame_chunk(frame_id)
            
            frames = [torch.tensor(self.frame_cache.get(fid, np.zeros((224, 224, 3), dtype=np.float32)).transpose(2, 0, 1))
                      for fid in range(frame_id, end_frame_id, self.stride)]
            frames = torch.stack(frames) if frames else torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            
            roles = [self._encode_roles(self.frame_to_chars.get(fid, set())) for fid in range(frame_id, end_frame_id, self.stride)]
            roles = torch.stack(roles) if roles else torch.zeros((1, self.role_dim), dtype=torch.float32)
            
            flow_value = self.flow_cache.get((timestamp, end_timestamp), 0.0)
            if flow_value > 5.0:
                label = 1
            
            frames = self.strong_aug(frames) if label == 1 else self.weak_aug(frames)
            return frames, roles, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            logging.error(f"獲取樣本 {idx} 失敗: {e}")
            return torch.zeros((1, 3, 224, 224)), torch.zeros((1, self.role_dim)), torch.tensor(0, dtype=torch.float32)

class MLSceneSegmenter(nn.Module):
    def __init__(self, video_path, frame_to_chars, role_dim, fps=30, k=60, hidden_dim=128):
        super().__init__()
        self.video_path = video_path
        self.frame_to_chars = frame_to_chars
        self.fps = fps
        self.k = k
        self.visual_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.role_embed = nn.Linear(max(role_dim, 1), hidden_dim)
        self.lstm = nn.LSTM(768 + hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.device = torch.device("cpu")
        self.to(self.device)
        # 移除 torch.compile，改用 eager 模式
        logging.info(f"初始化 MLSceneSegmenter，角色維度: {role_dim}")

    def forward(self, frames, roles):
        batch_size, num_frames, _, h, w = frames.size()
        frames = frames.view(batch_size * num_frames, 3, h, w)
        vis_feats = self.visual_extractor(frames).last_hidden_state[:, 0, :].view(batch_size, num_frames, 768).mean(dim=1)
        roles = roles.view(batch_size * num_frames, -1)
        role_embs = self.role_embed(roles).view(batch_size, num_frames, -1).mean(dim=1)
        combined = torch.cat([vis_feats, role_embs], dim=-1).unsqueeze(1)
        lstm_out, _ = self.lstm(combined)
        return self.classifier(lstm_out.squeeze(1)).squeeze(-1)

    def train_model(self, ground_truth_csv, segment_idx):
        try:
            dataset = SceneDataset(self.video_path, self.frame_to_chars, ground_truth_csv, self.fps, self.k, segment_idx=segment_idx)
            train_size = int(0.8 * len(dataset))
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=multiprocessing.cpu_count())
            val_loader = DataLoader(val_dataset, batch_size=8, num_workers=multiprocessing.cpu_count())
            
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
            criterion = nn.BCELoss()
            best_f1, stagnant_epochs = 0, 0
            for epoch in range(3):
                self.train()
                train_loss = 0
                for frames, roles, label in tqdm(train_loader, desc=f"訓練 Epoch {epoch+1}（分段 {segment_idx}）"):
                    frames, roles, label = frames.to(self.device), roles.to(self.device), label.to(self.device)
                    optimizer.zero_grad()
                    output = self(frames, roles)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                logging.info(f"Epoch {epoch+1} 訓練損失（分段 {segment_idx}）: {train_loss / len(train_loader):.4f}")
                
                self.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for frames, roles, label in val_loader:
                        frames, roles, label = frames.to(self.device), roles.to(self.device), label.to(self.device)
                        output = self(frames, roles)
                        val_preds.extend((output > 0.5).float().cpu().numpy())
                        val_labels.extend(label.cpu().numpy())
                f1 = sum((np.array(val_preds) == np.array(val_labels)) & (np.array(val_labels) == 1)) / (
                    sum(np.array(val_preds) == 1) + 1e-10
                )
                logging.info(f"Epoch {epoch+1} 驗證 F1（分段 {segment_idx}）: {f1:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    stagnant_epochs = 0
                    torch.save(self.state_dict(), os.path.join(r"C:\Users\user\Desktop\output", f"best_model_segment_{segment_idx}.pth"))
                else:
                    stagnant_epochs += 1
                    if stagnant_epochs >= 2:
                        logging.info(f"分段 {segment_idx} 提前停止於 Epoch {epoch+1}")
                        break
        except Exception as e:
            logging.error(f"訓練失敗（分段 {segment_idx}）: {e}")
            raise

    def analyze_video(self, ground_truth_csv, num_segments=5):
        try:
            total_frames = 5956
            frames_per_segment = total_frames // num_segments
            all_probs, all_timestamps = [], []
            
            for segment_idx in range(num_segments):
                start_frame = segment_idx * frames_per_segment
                logging.info(f"分析分段 {segment_idx} (幀 {start_frame}-{start_frame + frames_per_segment - 1})")
                dataset = SceneDataset(self.video_path, self.frame_to_chars, ground_truth_csv, self.fps, self.k, segment_idx=segment_idx)
                if not dataset.samples:
                    logging.warning(f"分段 {segment_idx} 無樣本，跳過")
                    continue
                loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=multiprocessing.cpu_count())
                
                try:
                    self.load_state_dict(torch.load(os.path.join(r"C:\Users\user\Desktop\output", f"best_model_segment_{segment_idx}.pth")))
                except FileNotFoundError:
                    logging.warning(f"未找到模型權重 best_model_segment_{segment_idx}.pth，跳過")
                    continue
                
                self.eval()
                with torch.no_grad():
                    for i, (frames, roles, _) in enumerate(tqdm(loader, desc=f"分析分段 {segment_idx}")):
                        frames, roles = frames.to(self.device), roles.to(self.device)
                        probs = self(frames, roles).cpu().numpy()
                        all_probs.extend(probs if probs.ndim == 1 else [probs])
                        batch_start_idx = i * loader.batch_size
                        all_timestamps.extend(t for _, _, _, t, _ in dataset.samples[batch_start_idx:batch_start_idx + len(probs)])
            
            all_probs = np.array(all_probs)
            peaks, _ = find_peaks(all_probs, height=0.3, distance=int(self.fps * 5))
            logging.info(f"找到 {len(peaks)} 個峰值")
            segments = []
            prev_time = 0
            for peak_idx in peaks:
                if peak_idx < len(all_timestamps):
                    time = all_timestamps[peak_idx]
                    if time - prev_time >= 5:
                        segments.append({"start_time_seconds": prev_time, "end_time_seconds": time, "duration_seconds": time - prev_time})
                        prev_time = time
            if all_timestamps and prev_time < all_timestamps[-1]:
                segments.append({"start_time_seconds": prev_time, "end_time_seconds": all_timestamps[-1], "duration_seconds": all_timestamps[-1] - prev_time})
            elif not segments:
                segments.append({"start_time_seconds": 0, "end_time_seconds": all_timestamps[-1] if all_timestamps else 0, "duration_seconds": all_timestamps[-1] if all_timestamps else 0})
            
            segment_df = pd.DataFrame(segments)
            segment_df.to_csv(r"C:\Users\user\Desktop\output\scene_segments.csv", index=False)
            
            with open(r"C:\Users\user\Desktop\output\scene_summary.txt", "w", encoding="utf-8") as f:
                f.write(f"總場景數: {len(segments)}\n")
                f.write(f"總影片時長: {all_timestamps[-1] if all_timestamps else 0:.2f} 秒\n")
                for i, seg in enumerate(segments):
                    chars = set()
                    for frame_id in range(int(seg["start_time_seconds"] * self.fps), int(seg["end_time_seconds"] * self.fps)):
                        chars.update(self.frame_to_chars.get(frame_id, set()))
                    f.write(f"場景 {i+1}: {seg['start_time_seconds']:.2f} - {seg['end_time_seconds']:.2f} 秒，持續 {seg['duration_seconds']:.2f} 秒，角色: {', '.join(sorted(chars))}\n")
            logging.info(f"生成場景摘要，包含 {len(segments)} 個場景")
        except Exception as e:
            logging.error(f"影片分析失敗: {e}")
            raise

def correct_ground_truth_csv(ground_truth_csv, output_csv):
    df = pd.read_csv(ground_truth_csv)
    df['start_time_seconds'] = pd.to_datetime(df['start_time'], errors='coerce').apply(
        lambda x: (x.timestamp() - pd.Timestamp("1900-01-01").timestamp()) if pd.notna(x) else 0.0
    )
    df['end_time_seconds'] = pd.to_datetime(df['end_time'], errors='coerce').apply(
        lambda x: (x.timestamp() - pd.Timestamp("1900-01-01").timestamp()) if pd.notna(x) else 0.0
    )
    df['duration_seconds'] = df['end_time_seconds'] - df['start_time_seconds']
    df = df[df['duration_seconds'] > 0].sort_values('start_time_seconds')
    df.to_csv(output_csv, index=False)
    logging.info(f"已修正 ground_truth.csv，保存至 {output_csv}")

def main():
    try:
        # CPU 加速設置（移除 OMP_PROC_BIND 以避免警告）
        num_cores = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
        os.environ['OMP_SCHEDULE'] = 'STATIC'
        # 移除 OMP_PROC_BIND，因為 KMP_AFFINITY 已涵蓋
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        os.environ['KMP_BLOCKTIME'] = '1'
        # 禁用 torch.compile 錯誤（若已修復）
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        video_path = r"C:\Users\user\Desktop\output\sample\復仇者聯盟 終局之戰.Avengers.Endgame.2019.mp4"
        character_csv_path = r"C:\Users\user\Desktop\output\復仇者聯盟 終局之戰.Avengers.Endgame.2019\csv_files\character_features_復仇者聯盟 終局之戰.Avengers.Endgame.2019.csv"
        role_id_map_path = r"C:\Users\user\Desktop\output\復仇者聯盟 終局之戰.Avengers.Endgame.2019\role_id_map_復仇者聯盟 終局之戰.Avengers.Endgame.2019.json"
        ground_truth_csv = r"C:\Users\user\Desktop\output\ground_truth.csv"
        output_ground_truth_csv = r"C:\Users\user\Desktop\output\ground_truth_corrected.csv"
        
        for path in [video_path, character_csv_path, role_id_map_path, ground_truth_csv]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"檔案不存在: {path}")
        
        if not os.path.exists(output_ground_truth_csv):
            correct_ground_truth_csv(ground_truth_csv, output_ground_truth_csv)
        
        character_df = pd.read_csv(character_csv_path)
        unique_chars = sorted(set(character_df['CharacterName'].unique()) - {'未知角色'})
        logging.info(f"Character CSV 唯一角色數: {len(unique_chars)}")
        
        frame_to_chars = {}
        for frame_id, group in character_df.groupby("FrameID"):
            chars = set(group["CharacterName"]) - {'未知角色'}
            if chars:
                frame_to_chars[frame_id] = chars
        for frame_id in range(5956):
            frame_to_chars.setdefault(frame_id, set())
        logging.info(f"frame_to_chars 初始化完成，影格數: {len(frame_to_chars)}")
        
        if os.path.exists(role_id_map_path):
            with open(role_id_map_path, 'r', encoding='utf-8') as f:
                role_id_map = json.load(f)
            if set(role_id_map.values()) - {'未知角色'} != set(unique_chars):
                role_id_map = {f"ID_{i}": char for i, char in enumerate(unique_chars)}
                with open(role_id_map_path, 'w', encoding='utf-8') as f:
                    json.dump(role_id_map, f, ensure_ascii=False, indent=4)
                logging.info(f"更新 role_id_map.json")
        
        model = MLSceneSegmenter(video_path, frame_to_chars, role_dim=len(unique_chars), fps=30, k=60)
        for segment_idx in range(5):
            model.train_model(output_ground_truth_csv, segment_idx)
        model.analyze_video(output_ground_truth_csv, num_segments=5)
    except Exception as e:
        logging.error(f"程式執行失敗: {e}")
        raise

if __name__ == "__main__":
    main()