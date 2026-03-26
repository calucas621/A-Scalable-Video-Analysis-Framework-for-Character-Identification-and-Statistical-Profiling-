import pandas as pd
import logging
import os

def evaluate_accuracy(character_csv_path, output_dir):
    # 設置日誌
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # 讀取 CSV
        df = pd.read_csv(character_csv_path)
        logging.info(f"載入 CSV 檔案: {character_csv_path}, 總筆數: {len(df)}")
        
        # 計算未知角色筆數 (CharacterName == "未知角色")
        unknown_frames = len(df[df['CharacterName'] == "未知角色"])
        total_frames = len(df)
        
        # 計算準確率
        accuracy = (total_frames - unknown_frames) / total_frames * 100 if total_frames > 0 else 0
        
        # 生成輸出檔案
        accuracy_file = os.path.join(output_dir, "accuracy.txt")
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            f.write(f"識別準確率: {accuracy:.2f}%\n")
            f.write(f"總影格數: {total_frames}\n")
            f.write(f"未識別影格數: {unknown_frames}\n")
        
        logging.info(f"準確率檔案已生成: {accuracy_file}")
    except Exception as e:
        logging.error(f"評估準確率失敗: {e}")

# 輸入檔案路徑
character_csv_path = r"C:\Users\user\Desktop\output\34_獅子王\csv_files\character_features_34_獅子王.csv"
output_dir = os.path.dirname(character_csv_path)  # 輸出到同目錄

# 執行評估
evaluate_accuracy(character_csv_path, output_dir)