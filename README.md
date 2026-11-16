# Baseball Ensemble Learning｜棒球球員薪資預測比較（公開版）

本專案為輔仁大學 115-1 機器學習課程作業，使用 R 語言比較 **CART Regression Tree**、**Random Forest** 與 **XGBoost** 三種模型的棒球球員對數薪資（`Logsalary`）預測誤差，並以 **Multiple Linear Regression (`glm`)** 作為傳統統計基準模型。

---

## 一、資料來源與公開範圍說明

1. **資料來源**：本專案分析使用課程教師提供的棒球球員薪資資料集（`ball.txt`，263 筆觀測值、26 個欄位）。
2. **未納入公開 Repository 之檔案**：基於資料公開範圍與課程教材授權考量，本公開版 Repository **不包含**以下檔案：
   - 原始資料集：`ball.txt`
   - 教師課堂範例程式：`CART_Baseball.r`、`CART_Test.r`、`RF_Test.r`、`XGBoost_Test.r`
   - 含逐筆球員真實／預測數值與模型物件之本地產出：`outputs/tables/test_predictions.csv`、`outputs/models_workspace.rds`
3. **數據真實性聲明**：本公開版所附之彙總比較表（[outputs/tables/model_comparison.csv](outputs/tables/model_comparison.csv)）與三張視覺化圖表（[outputs/figures/](outputs/figures/)），均來自先前以完整合法資料集實際執行本專案 R 程式後所產出之真實結果，並非模擬或虛構數據。

---

## 二、專案目錄架構

```text
Baseball_Ensemble_Learning/
├── run_all.R                       # [主程式] 依序執行前處理、四模型訓練、測試集評估與繪圖
├── R/
│   ├── 00_setup.R                  # 套件檢查／自動安裝與輸出目錄建立
│   ├── 01_data_prep.R              # 檢查 ball.txt、排除 Id/Name/Salary、選取 19 特徵與 80/20 切分
│   ├── 02_train_models.R           # 僅使用訓練集訓練 Linear Regression / CART / RF / XGBoost
│   └── 03_evaluate_and_plot.R      # 測試集預測、計算 MSE / RMSE 與產出表格及圖表
├── tests/
│   └── test_pipeline.R             # 獨立自動化驗證腳本（檢查公開產出；若有 ball.txt 則完整重算驗算）
├── outputs/
│   ├── tables/
│   │   └── model_comparison.csv    # 四模型訓練集與測試集 MSE / RMSE 比較彙總表
│   └── figures/
│       ├── 01_cart_tree.png        # CART 決策樹結構圖
│       ├── 02_pred_vs_obs.png      # 四模型 Predicted vs. Observed 散佈對照圖
│       └── 03_mse_rmse_bar.png     # 四模型測試集 MSE 與 RMSE 長條比較圖
├── docs/
│   └── report.md                   # 繁體中文研究結果分析報告
├── .gitignore                      # Git 忽略規則（含原始資料與敏感產物排除保護）
└── README.md                       # 本專案說明與重現指南
```

---

## 三、資料處理與公平比較設計

- **排除欄位（防目標洩漏）**：嚴格移除 `Id`、`Name`、`Salary`（`Logsalary = ln(Salary)`）。
- **預測目標**：`Logsalary`。
- **共同預測特徵**：採用 `fit1` 的 19 個特徵（排除高基數且含單一樣本類別之 `Team86`、`Pos86`、`Team87`），四模型使用完全等價之特徵資訊。
- **資料切分**：設定 `set.seed(123)`，使用 `caTools::sample.split(model_df$Logsalary, SplitRatio = 0.8)` 切分為 **210 筆訓練集（80%）** 與 **53 筆測試集（20%）**，四模型共用相同樣本。

---

## 四、環境需求與重現步驟

### 4.1 環境需求

- **R**：建議 R 4.4 以上（本專案於 R 4.6.1 實測通過）
- **使用套件**：`rpart`、`rpart.plot`、`caTools`、`randomForest`、`xgboost`、`ggplot2`、`gridExtra`

### 4.2 如何在本地重現分析

1. **取得並放置合法資料副本**：請先自行向課程或合法來源取得 `ball.txt` 副本，並放置於本專案根目錄下（即 `./ball.txt`，該檔案已列於 `.gitignore`，不會被意外納入版本追蹤）。
2. **切換至專案根目錄並安裝／載入套件**：執行 `run_all.R` 時，`R/00_setup.R` 會自動檢查上述必要 R 套件並補齊缺少的 CRAN 套件；亦可先於 R 中手動執行 `install.packages(c("rpart", "rpart.plot", "caTools", "randomForest", "xgboost", "ggplot2", "gridExtra"))`。
3. **執行完整分析與驗證**：

```bash
# 1. 執行完整分析流程（前處理、四模型訓練、測試集評估與圖表輸出）
Rscript --vanilla run_all.R

# 2. 執行獨立自動化驗證（檢查防資料洩漏與 MSE / RMSE 重算）
Rscript --vanilla tests/test_pipeline.R
```

*(註：若在未放入 `ball.txt` 的公開版目錄下直接執行 `Rscript --vanilla tests/test_pipeline.R`，腳本會自動驗證公開的 `model_comparison.csv` 與圖檔完整性，並提示須放入 `ball.txt` 後才能執行完整重訓驗算。)*

---

## 五、實驗結果摘要

於共同 53 筆測試集上之實際執行結果（詳細解讀請參閱 [docs/report.md](docs/report.md)）：

| 排名 | 模型名稱 | Train MSE | Train RMSE | Test MSE | Test RMSE |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Random Forest** | 0.0210 | 0.1449 | **0.1865** | **0.4318** |
| 2 | **XGBoost** | 0.0022 | 0.0465 | **0.2130** | **0.4615** |
| 3 | **CART** | 0.1084 | 0.3293 | **0.2184** | **0.4673** |
| 4 | **Linear Regression** *(基準)* | 0.2703 | 0.5199 | **0.3958** | **0.6291** |
