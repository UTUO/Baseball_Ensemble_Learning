# ==============================================================================
# tests/test_pipeline.R
# 獨立自動化驗證與誤差驗算腳本（公開版相容）
# 注意：本腳本僅讀取 outputs/ 產出物進行驗證，絕不呼叫 run_all.R，避免互相遞迴
# ==============================================================================

test_results <- list()

run_check <- function(name, expr) {
  ok <- tryCatch({
    res <- eval(substitute(expr), envir = parent.frame())
    isTRUE(res)
  }, error = function(e) {
    message(sprintf("[FAIL] %s -> 發生錯誤：%s", name, conditionMessage(e)))
    FALSE
  })
  if (ok) {
    message(sprintf("[PASS] %s", name))
  } else {
    stop(sprintf("驗證失敗：%s", name), call. = FALSE)
  }
  test_results[[name]] <<- ok
}

# ------------------------------------------------------------------------------
# 公開版基礎檢核：必要彙總表格、圖檔與四模型數值格式皆有效
# ------------------------------------------------------------------------------
run_check("1. 公開版彙總結果表與三張視覺化圖檔均存在且非空", {
  public_outputs <- c(
    "outputs/tables/model_comparison.csv",
    "outputs/figures/01_cart_tree.png",
    "outputs/figures/02_pred_vs_obs.png",
    "outputs/figures/03_mse_rmse_bar.png"
  )
  all(file.exists(public_outputs)) && all(file.info(public_outputs)$size > 0)
})

comp_df <- read.csv("outputs/tables/model_comparison.csv", stringsAsFactors = FALSE)

run_check("2. model_comparison.csv 包含四個模型且所有 MSE / RMSE 指標皆為有效有限數值", {
  expected_models <- c("Linear Regression", "CART", "Random Forest", "XGBoost")
  expected_cols <- c("Model", "Train_MSE", "Train_RMSE", "Test_MSE", "Test_RMSE", "Test_Rank")
  metric_cols <- c("Train_MSE", "Train_RMSE", "Test_MSE", "Test_RMSE", "Test_Rank")

  cols_ok <- all(expected_cols %in% colnames(comp_df))
  models_ok <- setequal(comp_df$Model, expected_models) && nrow(comp_df) == 4L
  finite_ok <- all(vapply(comp_df[, metric_cols], function(col) {
    is.numeric(col) && all(is.finite(col)) && all(col > 0)
  }, logical(1)))

  rmse_consistent <- max(abs(sqrt(comp_df$Test_MSE) - comp_df$Test_RMSE)) < 1e-10 &&
    max(abs(sqrt(comp_df$Train_MSE) - comp_df$Train_RMSE)) < 1e-10

  cols_ok && models_ok && finite_ok && rmse_consistent
})

# ------------------------------------------------------------------------------
# 完整本地執行後之進階檢核（需已放入 ball.txt 並執行 run_all.R）
# ------------------------------------------------------------------------------
has_full_local_artifacts <- file.exists("ball.txt") &&
  file.exists("outputs/models_workspace.rds") &&
  file.exists("outputs/tables/test_predictions.csv")

if (has_full_local_artifacts) {
  suppressMessages(source("R/00_setup.R"))

  ws <- readRDS("outputs/models_workspace.rds")
  pred_df <- read.csv("outputs/tables/test_predictions.csv", stringsAsFactors = FALSE)

  run_check("3. 四模型物件建立正確且測試集預測值數量與數值皆正常", {
    classes_ok <- inherits(ws$lm_fit, "glm") &&
      inherits(ws$cart_fit, "rpart") &&
      inherits(ws$rf_fit, "randomForest") &&
      inherits(ws$xgb_fit, "xgb.Booster")

    dtest_chk <- xgboost::xgb.DMatrix(data = ws$x_test)
    p_lm   <- as.numeric(predict(ws$lm_fit, newdata = ws$test_data))
    p_cart <- as.numeric(predict(ws$cart_fit, newdata = ws$test_data))
    p_rf   <- as.numeric(predict(ws$rf_fit, newdata = ws$test_data))
    p_xgb  <- as.numeric(predict(ws$xgb_fit, newdata = dtest_chk))

    n_test <- nrow(ws$test_data)
    preds <- list(p_lm, p_cart, p_rf, p_xgb)
    lengths_ok <- all(vapply(preds, length, integer(1)) == n_test)
    finite_ok  <- all(vapply(preds, function(v) all(is.finite(v)), logical(1)))

    csv_match <- max(abs(p_lm - pred_df$pred_LM)) < 1e-10 &&
      max(abs(p_cart - pred_df$pred_CART)) < 1e-10 &&
      max(abs(p_rf - pred_df$pred_RF)) < 1e-10 &&
      max(abs(p_xgb - pred_df$pred_XGB)) < 1e-10

    classes_ok && lengths_ok && finite_ok && csv_match
  })

  run_check("4. Id / Name / Salary 已徹底排除於資料與所有模型之外", {
    forbidden <- c("Id", "Name", "Salary", "Team86", "Pos86", "Team87")

    data_clean <- !any(forbidden %in% colnames(ws$train_data)) &&
      !any(forbidden %in% colnames(ws$test_data)) &&
      !any(forbidden %in% colnames(ws$x_train)) &&
      !any(forbidden %in% colnames(ws$x_test))

    target_ok <- ("Logsalary" %in% colnames(ws$train_data)) &&
      ("Logsalary" %in% colnames(ws$test_data)) &&
      !("Logsalary" %in% colnames(ws$x_train)) &&
      !("Logsalary" %in% colnames(ws$x_test))

    lm_terms   <- attr(stats::terms(ws$lm_fit), "term.labels")
    cart_terms <- attr(stats::terms(ws$cart_fit), "term.labels")
    rf_terms   <- rownames(ws$rf_fit$importance)
    xgb_terms  <- xgboost::getinfo(ws$xgb_fit, "feature_name")

    models_clean <- !any(forbidden %in% lm_terms) &&
      !any(forbidden %in% cart_terms) &&
      !any(forbidden %in% rf_terms) &&
      !any(forbidden %in% xgb_terms) &&
      !("Logsalary" %in% lm_terms) &&
      !("Logsalary" %in% cart_terms) &&
      !("Logsalary" %in% rf_terms) &&
      !("Logsalary" %in% xgb_terms)

    dims_ok <- (ncol(ws$train_data) == 20L) &&
      (ncol(ws$test_data) == 20L) &&
      (ncol(ws$x_train) == 19L) &&
      (ncol(ws$x_test) == 19L) &&
      (length(lm_terms) == 19L) &&
      (length(rf_terms) == 19L) &&
      (length(xgb_terms) == 19L)

    data_clean && target_ok && models_clean && dims_ok
  })

  run_check("5. 80/20 切分比例、互斥索引與真實反應變數皆完全一致", {
    raw_chk <- read.table("ball.txt", header = TRUE, sep = "", quote = "\"", dec = ".", stringsAsFactors = FALSE)

    counts_ok <- (length(ws$train_idx) == 210L) &&
      (length(ws$test_idx) == 53L) &&
      (length(ws$train_idx) + length(ws$test_idx) == nrow(raw_chk)) &&
      (nrow(ws$train_data) == 210L) &&
      (nrow(ws$test_data) == 53L)

    disjoint_ok <- length(intersect(ws$train_idx, ws$test_idx)) == 0L

    y_match_ok <- identical(as.integer(ws$test_idx), as.integer(pred_df$Row_Index)) &&
      max(abs(ws$y_test - raw_chk$Logsalary[ws$test_idx])) < 1e-12 &&
      max(abs(pred_df$true_Logsalary - raw_chk$Logsalary[ws$test_idx])) < 1e-12

    counts_ok && disjoint_ok && y_match_ok
  })

  run_check("6. 測試集 MSE 與 RMSE 獨立算術驗算與報表完全一致", {
    y_true <- pred_df$true_Logsalary
    pred_cols <- c(
      "Linear Regression" = "pred_LM",
      "CART"              = "pred_CART",
      "Random Forest"     = "pred_RF",
      "XGBoost"           = "pred_XGB"
    )

    all_match <- TRUE
    for (m_name in names(pred_cols)) {
      p_vec <- pred_df[[pred_cols[[m_name]]]]
      recalc_mse  <- sum((p_vec - y_true)^2) / length(y_true)
      recalc_rmse <- sqrt(recalc_mse)

      row_idx <- which(comp_df$Model == m_name)
      if (length(row_idx) != 1L) {
        all_match <- FALSE
        break
      }
      diff_mse  <- abs(recalc_mse - comp_df$Test_MSE[row_idx])
      diff_rmse <- abs(recalc_rmse - comp_df$Test_RMSE[row_idx])
      if (diff_mse >= 1e-10 || diff_rmse >= 1e-10) {
        all_match <- FALSE
        break
      }
    }
    all_match
  })
} else {
  message("[INFO] 未偵測到 ball.txt 與完整本地模型工作區（models_workspace.rds / test_predictions.csv）。")
  message("[INFO] 本公開版未附帶原始資料；若要執行完整模型重訓與逐筆驗算，請先取得合法 'ball.txt' 置於根目錄並執行 `Rscript --vanilla run_all.R`。")
}

cat("\n==================== 自動化驗證總結 ====================\n")
cat(sprintf("已執行之 %d 項檢核全數通過 (PASS)。\n", length(test_results)))
cat("========================================================\n")
