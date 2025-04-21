# 文本情緒分析：Word2Vec vs. BERT 模型比較

## 專案簡介

本專案旨在探討與比較兩種常見的自然語言處理（NLP）模型——Word2Vec（結合 LSTM）與 BERT——在文本情緒分析任務上的表現。我們使用了 Kaggle Sentiment140 資料集進行訓練與評估。


## 資料集

* **名稱：** [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140)
* **大小：** 160 萬筆 Twitter 推文資料。
* **欄位：** 主要使用 `text`（推文內容）與 `target`（情緒標籤：0=負面, 4=正面）。
* **資料預處理：**
    * 移除非必要欄位 (`Date`, `ids`, `flag`, `user`)。
    * 文本清理（例如：移除 URL、@使用者名稱、多餘字元、HTML 標籤等）。
    * 將情緒標籤轉換為數值（0 和 1）。
    * （Word2Vec 使用）將文本轉換為詞向量序列。
    * （BERT 使用）利用 TensorFlow Hub 的預處理層進行 Tokenization 與格式轉換。

## 模型架構與訓練

### 1. Word2Vec + LSTM

* **詞嵌入：** 使用 `gensim` 的 Word2Vec 將詞彙轉換為 100 維向量。
* **模型層：**
    1.  `SpatialDropout1D` (rate=0.4)：防止過擬合。
    2.  `LSTM` (units=196)：提取序列特徵。
    3.  `Dense` (units=10, activation='relu')：特徵轉換。
    4.  `Dense` (units=2, activation='softmax', kernel_regularizer=L2(0.01))：輸出層，進行二分類，加入 L2 正則化。
* **訓練細節：**
    * **資料筆數：** 160 萬
    * **優化器：** Adam
    * **損失函數：** Categorical Crossentropy
    * **評估指標：** Accuracy
    * **Epochs：** 5
    * **Batch Size：** 128
    * **Early Stopping：** 監控 `val_loss`，若連續 3 個 epoch 未改善則停止。
    * **訓練時間：** 約 15 分鐘 (本機)
    * **記憶體用量：** 約 92 GB

### 2. BERT

* **模型：** 使用 TensorFlow Hub 預訓練的 BERT 模型 (`bert_en_uncased_L-12_H-768_A-12/3`) 與對應的預處理層 (`bert_en_uncased_preprocess/3`)。
* **模型層：**
    1.  `Input Layer` (dtype=tf.string)：接收文本輸入。
    2.  `Preprocessing Layer` (TF Hub)：文本預處理。
    3.  `BERT Encoder Layer` (TF Hub, trainable=True)：提取文本特徵，設定為可微調。
    4.  `Dropout` (rate=0.1)：防止過擬合。
    5.  `Dense` (units=1)：輸出層，用於二分類。
* **訓練細節：**
    * **資料筆數：** 4 萬 (因 Colab 環境限制而減少)
    * **優化器：** Adam (Learning Rate = 3e-5)
    * **損失函數：** Binary Crossentropy (from_logits=True)
    * **評估指標：** Binary Accuracy
    * **Epochs：** 3 (因環境限制)
    * **Batch Size：** 32
    * **Early Stopping：** 監控 `val_loss`，若連續 2 個 epoch 未改善則停止。
    * **訓練時間：** 約 70 分鐘 (Colab)

## 結果比較

| 模型     | 資料筆數 | 訓練時間         | 測試集準確率 (Accuracy) |
| :------- | :------- | :--------------- | :---------------------- |
| Word2Vec | 160 萬   | ~15 分鐘 (本機)  | 77.78%                  |
| BERT     | 4 萬     | ~70 分鐘 (Colab) | 78.45%                  |

**觀察：**

* **Word2Vec + LSTM：**
    * 訓練速度快，記憶體需求高。
    * 在訓練過程中，驗證集的準確率和損失有波動，但整體趨勢向好。
    * 適合相對簡單的 NLP 任務，性價比較高。
* **BERT：**
    * 即使使用遠少於 Word2Vec 的資料量，仍達到略高的準確率。
    * 訓練時間較長（未包含 BERT 預訓練時間）。
    * 訓練集表現持續改善，但測試集損失上升，可能出現過擬合（或因測試集資料量少）。
    * 適合處理更複雜的 NLP 任務，能捕捉更豐富的上下文資訊。

## 結論

Word2Vec 結合 LSTM 在此文本情緒分析任務上已能達到不錯的表現，且訓練時間成本相對較低。BERT 模型雖然需要更長的訓練時間（且受限於計算資源，本次僅使用部分資料），但在少量資料下展現了更高的準確率潛力，顯示其在理解複雜語義上的優勢。

選擇何種模型取決於具體的應用場景、可用的計算資源、以及對準確率的要求。本次專案也突顯了進行 NLP 模型訓練時，在時間與硬體資源（尤其是記憶體）上的投入是相當可觀的。
