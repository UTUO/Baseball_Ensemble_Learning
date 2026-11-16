# ==============================================================================
# 01_data_prep.R
# 讀取原始資料、排除指定欄位與高基數類別欄位、統一型別並執行 80/20 資料切分
# ==============================================================================

# 0. 檢查原始資料檔是否存在（公開版未納入原始資料，需由使用者自行取得並放置於根目錄）
if (!file.exists("ball.txt")) {
  stop(
    "找不到原始資料檔 'ball.txt'。本公開版 Repository 未包含課程原始資料，",
    "請先取得合法的 'ball.txt' 副本並放置於專案根目錄後，再執行 run_all.R。",
    call. = FALSE
  )
}

# 1. 讀取原始棒球資料（唯讀，不更動原始 ball.txt）
raw_df <- read.table(
  "ball.txt",
  header = TRUE,
  sep = "",
  quote = "\"",
  dec = ".",
  stringsAsFactors = FALSE
)

# 2. 明確定義必須排除的欄位與共同 19 個預測特徵（對應課程範例 fit1）
# - Id、Name 為球員識別碼，不具備預測意義
# - Salary 與 Logsalary = log(Salary) 存在完全數學對應，納入將造成目標洩漏
# - Team86、Pos86、Team87 因高基數與測試集未見類別 / 共線性問題予以排除
excluded_cols <- c("Id", "Name", "Salary")
excluded_high_card_cols <- c("Team86", "Pos86", "Team87")
target_col <- "Logsalary"

feature_cols <- c(
  "Bat86", "Hit86", "Hr86", "Run86", "Rb86", "Wlk86",
  "Yrs", "Batcr", "Hitcr", "Hrcr", "Runcr", "Rbcr", "Wlkcr",
  "League86", "Div86", "Puto86", "Asst86", "Err86", "League87"
)

# 3. 建構模型專用資料框，僅保留目標變數與 19 個預測特徵
model_df <- raw_df[, c(target_col, feature_cols), drop = FALSE]

# 強制檢查排除欄位與缺漏值，確保無資料洩漏與資料異常
stopifnot(
  !any(excluded_cols %in% colnames(model_df)),
  !any(excluded_high_card_cols %in% colnames(model_df)),
  identical(colnames(model_df), c(target_col, feature_cols)),
  sum(is.na(model_df)) == 0
)

# 4. 將 3 個二元類別變數轉為固定水準之 factor（確保 randomForest 與預測一致）
model_df$League86 <- factor(model_df$League86, levels = c("A", "N"))
model_df$Div86    <- factor(model_df$Div86, levels = c("E", "W"))
model_df$League87 <- factor(model_df$League87, levels = c("A", "N"))

# 5. 設定隨機種子並依目標變數 Logsalary 進行 80% 訓練 / 20% 測試切分
set.seed(123)
split_mask <- caTools::sample.split(model_df$Logsalary, SplitRatio = 0.8)

train_idx <- which(split_mask == TRUE)
test_idx  <- which(split_mask == FALSE)

train_data <- model_df[train_idx, , drop = FALSE]
test_data  <- model_df[test_idx, , drop = FALSE]

# 6. 為 XGBoost 建立數值設計矩陣
# 使用不含截距欄位之 model.matrix(..., )[, -1]，使 3 個二元因子各自對應 1 個 0/1 指標欄位，
# 維持精確 19 個輸入維度，與其他模型資訊完全等價。
x_train <- model.matrix(Logsalary ~ ., data = train_data)[, -1, drop = FALSE]
x_test  <- model.matrix(Logsalary ~ ., data = test_data)[, -1, drop = FALSE]

y_train <- train_data$Logsalary
y_test  <- test_data$Logsalary

message(sprintf(
  "[Data Prep] 總樣本數：%d | 訓練集：%d (%.1f%%) | 測試集：%d (%.1f%%) | 特徵數：%d",
  nrow(model_df),
  nrow(train_data), 100 * nrow(train_data) / nrow(model_df),
  nrow(test_data), 100 * nrow(test_data) / nrow(model_df),
  length(feature_cols)
))
