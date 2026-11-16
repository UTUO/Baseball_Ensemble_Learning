# ==============================================================================
# 00_setup.R
# 環境初始化、套件檢查安裝與輸出目錄建立
# ==============================================================================

# 設定 CRAN 鏡像來源，確保在非互動式 Rscript 環境下可自動下載安裝套件
if (is.null(getOption("repos")) || identical(getOption("repos")[["CRAN"]], "@CRAN@")) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

# 檢查使用者套件庫路徑；若系統套件庫不可寫入，優先設定並建立使用者套件庫
user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib)) {
  if (!dir.exists(user_lib)) {
    suppressWarnings(dir.create(user_lib, recursive = TRUE, showWarnings = FALSE))
  }
  if (dir.exists(user_lib)) {
    .libPaths(unique(c(user_lib, .libPaths())))
  }
}

# 若主要套件庫仍無寫入權限，於專案內建立本地套件庫目錄作為備援（已列於 .gitignore）
local_lib <- file.path(getwd(), ".R_libs")
if (dir.exists(local_lib)) {
  .libPaths(unique(c(local_lib, .libPaths())))
}

# 本研究必備之 R 套件清單（與老師範例程式採用相同核心套件）
required_pkgs <- c(
  "rpart",        # CART 迴歸決策樹
  "rpart.plot",   # CART 決策樹視覺化
  "caTools",      # 80/20 分層資料切分 (sample.split)
  "randomForest", # Random Forest 迴歸模型
  "xgboost",      # XGBoost 梯度提升迴歸模型
  "ggplot2",      # 繪製預測散佈圖與誤差比較圖
  "gridExtra"     # 多圖並排輸出 (grid.arrange)
)

# 檢查並自動安裝缺少的套件
is_installed <- vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)
missing_pkgs <- required_pkgs[!is_installed]

if (length(missing_pkgs) > 0) {
  message("偵測到缺少套件，開始安裝：", paste(missing_pkgs, collapse = ", "))
  target_lib <- .libPaths()[1]
  if (file.access(target_lib, 2) != 0) {
    dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
    .libPaths(unique(c(local_lib, .libPaths())))
    target_lib <- local_lib
  }
  install.packages(missing_pkgs, lib = target_lib, repos = "https://cloud.r-project.org")
}

# 確認所有套件皆已成功載入
still_missing <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(still_missing) > 0) {
  stop("以下套件安裝或載入失敗，請檢查網路與 R 環境：", paste(still_missing, collapse = ", "))
}

suppressPackageStartupMessages({
  for (pkg in required_pkgs) {
    library(pkg, character.only = TRUE)
  }
})

# 建立輸出資料夾結構
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

message(sprintf("[Setup] R 版本：%s；所有必要套件已就緒。", R.version.string))
