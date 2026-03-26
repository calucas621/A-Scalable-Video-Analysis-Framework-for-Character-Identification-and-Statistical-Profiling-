import numpy as np
import pandas as pd
import os

input_path = r"C:\Users\user\Desktop\output\12_腦筋急轉彎2\Inside Out 2 Features\Inside Out 2_features.npz"
output_folder = r"C:\Users\user\Desktop\output\12_腦筋急轉彎2\音訊特徵"

data = np.load(input_path, allow_pickle=True)
print("NPZ 內容：", data.files)

for key in data.files:
    array = data[key]
    print(f"{key} 原始 shape: {array.shape}")

    # 如果維度 > 2 → 攤平成 (樣本數, 其他全部展開)
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
        print(f"{key} 攤平後 shape: {array.shape}")

    df = pd.DataFrame(array)

    output_path = os.path.join(output_folder, f"{key}.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"已輸出：{output_path}")

