# ==============================================================================
# 03_evaluate_and_plot.R
# 於共同測試集進行預測、計算 MSE / RMSE、匯出比較表與視覺化圖表
# ==============================================================================

# 計算誤差指標之輔助函數
calc_mse <- function(pred, actual) {
  mean((pred - actual)^2)
}

calc_rmse <- function(pred, actual) {
  sqrt(calc_mse(pred, actual))
}

# 1. 訓練集預測（供參考對照過擬合程度）
pred_train_lm   <- as.numeric(predict(lm_fit, newdata = train_data))
pred_train_cart <- as.numeric(predict(cart_fit, newdata = train_data))
pred_train_rf   <- as.numeric(predict(rf_fit, newdata = train_data))
pred_train_xgb  <- as.numeric(predict(xgb_fit, newdata = dtrain))

# 2. 共同測試集預測（唯一正式評估依據）
pred_lm   <- as.numeric(predict(lm_fit, newdata = test_data))
pred_cart <- as.numeric(predict(cart_fit, newdata = test_data))
pred_rf   <- as.numeric(predict(rf_fit, newdata = test_data))
pred_xgb  <- as.numeric(predict(xgb_fit, newdata = dtest))

# 3. 整理四模型比較表
comparison_df <- data.frame(
  Model = c("Linear Regression", "CART", "Random Forest", "XGBoost"),
  Train_MSE = c(
    calc_mse(pred_train_lm, y_train),
    calc_mse(pred_train_cart, y_train),
    calc_mse(pred_train_rf, y_train),
    calc_mse(pred_train_xgb, y_train)
  ),
  Train_RMSE = c(
    calc_rmse(pred_train_lm, y_train),
    calc_rmse(pred_train_cart, y_train),
    calc_rmse(pred_train_rf, y_train),
    calc_rmse(pred_train_xgb, y_train)
  ),
  Test_MSE = c(
    calc_mse(pred_lm, y_test),
    calc_mse(pred_cart, y_test),
    calc_mse(pred_rf, y_test),
    calc_mse(pred_xgb, y_test)
  ),
  Test_RMSE = c(
    calc_rmse(pred_lm, y_test),
    calc_rmse(pred_cart, y_test),
    calc_rmse(pred_rf, y_test),
    calc_rmse(pred_xgb, y_test)
  ),
  stringsAsFactors = FALSE
)

comparison_df$Test_Rank <- rank(comparison_df$Test_MSE, ties.method = "min")

write.csv(comparison_df, "outputs/tables/model_comparison.csv", row.names = FALSE)

# 4. 整理測試集逐筆預測與平方誤差明細表（供後續獨立驗算使用）
predictions_df <- data.frame(
  Row_Index = test_idx,
  true_Logsalary = y_test,
  pred_LM = pred_lm,
  pred_CART = pred_cart,
  pred_RF = pred_rf,
  pred_XGB = pred_xgb,
  sqerr_LM = (pred_lm - y_test)^2,
  sqerr_CART = (pred_cart - y_test)^2,
  sqerr_RF = (pred_rf - y_test)^2,
  sqerr_XGB = (pred_xgb - y_test)^2
)

write.csv(predictions_df, "outputs/tables/test_predictions.csv", row.names = FALSE)

# 5. 儲存工作區核心物件（供獨立測試腳本檢核特徵排除與模型結構）
saveRDS(
  list(
    feature_cols = feature_cols,
    excluded_cols = excluded_cols,
    excluded_high_card_cols = excluded_high_card_cols,
    train_idx = train_idx,
    test_idx = test_idx,
    train_data = train_data,
    test_data = test_data,
    x_train = x_train,
    x_test = x_test,
    y_train = y_train,
    y_test = y_test,
    lm_fit = lm_fit,
    cart_fit = cart_fit,
    rf_fit = rf_fit,
    xgb_fit = xgb_fit,
    comparison_df = comparison_df,
    predictions_df = predictions_df
  ),
  file = "outputs/models_workspace.rds"
)

# 6. 產出視覺化圖表
# 圖一：CART 決策樹結構圖
grDevices::png(
  filename = "outputs/figures/01_cart_tree.png",
  width = 1400,
  height = 900,
  res = 150
)
rpart.plot::rpart.plot(
  cart_fit,
  extra = 101,
  fallen.leaves = TRUE,
  main = "CART Regression Tree for Logsalary (Training Set, n = 210)"
)
grDevices::dev.off()

# 圖二：四模型 Predicted vs. Observed 散佈比較圖
all_vals <- c(y_test, pred_lm, pred_cart, pred_rf, pred_xgb)
axis_min <- floor(min(4.0, min(all_vals) - 0.1) * 10) / 10
axis_max <- ceiling(max(8.0, max(all_vals) + 0.1) * 10) / 10

make_pred_plot <- function(pred_vec, model_title, mse_val, rmse_val, point_color) {
  df_plot <- data.frame(Predicted = pred_vec, Observed = y_test)
  ggplot2::ggplot(df_plot, ggplot2::aes(x = Predicted, y = Observed)) +
    ggplot2::geom_point(color = point_color, alpha = 0.8, size = 2.2) +
    ggplot2::geom_abline(
      intercept = 0,
      slope = 1,
      color = "red",
      linetype = "dashed",
      linewidth = 0.8
    ) +
    ggplot2::coord_fixed(xlim = c(axis_min, axis_max), ylim = c(axis_min, axis_max)) +
    ggplot2::labs(
      title = model_title,
      subtitle = sprintf("Test MSE = %.4f | Test RMSE = %.4f", mse_val, rmse_val),
      x = "Predicted Logsalary",
      y = "Observed Logsalary"
    ) +
    ggplot2::theme_bw(base_size = 11)
}

p_lm <- make_pred_plot(
  pred_lm, "Linear Regression (Baseline)",
  comparison_df$Test_MSE[1], comparison_df$Test_RMSE[1], "#4B6EAF"
)
p_cart <- make_pred_plot(
  pred_cart, "CART Regression Tree",
  comparison_df$Test_MSE[2], comparison_df$Test_RMSE[2], "#D97706"
)
p_rf <- make_pred_plot(
  pred_rf, "Random Forest Regression",
  comparison_df$Test_MSE[3], comparison_df$Test_RMSE[3], "#15803D"
)
p_xgb <- make_pred_plot(
  pred_xgb, "XGBoost Regression",
  comparison_df$Test_MSE[4], comparison_df$Test_RMSE[4], "#7C3AED"
)

grDevices::png(
  filename = "outputs/figures/02_pred_vs_obs.png",
  width = 1500,
  height = 1350,
  res = 150
)
gridExtra::grid.arrange(p_lm, p_cart, p_rf, p_xgb, ncol = 2)
grDevices::dev.off()

# 圖三：四模型測試集 MSE 與 RMSE 比較長條圖
bar_df <- rbind(
  data.frame(
    Model = comparison_df$Model,
    Metric = "Test MSE",
    Value = comparison_df$Test_MSE,
    stringsAsFactors = FALSE
  ),
  data.frame(
    Model = comparison_df$Model,
    Metric = "Test RMSE",
    Value = comparison_df$Test_RMSE,
    stringsAsFactors = FALSE
  )
)

model_order <- comparison_df$Model[order(comparison_df$Test_MSE)]
bar_df$Model <- factor(bar_df$Model, levels = model_order)
bar_df$Metric <- factor(bar_df$Metric, levels = c("Test MSE", "Test RMSE"))

p_bar <- ggplot2::ggplot(bar_df, ggplot2::aes(x = Model, y = Value, fill = Model)) +
  ggplot2::geom_col(width = 0.62, show.legend = FALSE) +
  ggplot2::geom_text(
    ggplot2::aes(label = sprintf("%.4f", Value)),
    vjust = -0.4,
    size = 3.6
  ) +
  ggplot2::facet_wrap(~ Metric, scales = "free_y") +
  ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.16))) +
  ggplot2::scale_fill_manual(values = c(
    "Linear Regression" = "#4B6EAF",
    "CART" = "#D97706",
    "Random Forest" = "#15803D",
    "XGBoost" = "#7C3AED"
  )) +
  ggplot2::labs(
    title = "Test Set Prediction Error Comparison (Logsalary)",
    subtitle = "Ordered from lowest Test MSE (best) to highest Test MSE (n_test = 53)",
    x = NULL,
    y = "Error Value"
  ) +
  ggplot2::theme_bw(base_size = 11) +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 15, hjust = 1))

ggplot2::ggsave(
  filename = "outputs/figures/03_mse_rmse_bar.png",
  plot = p_bar,
  width = 9.5,
  height = 4.8,
  dpi = 150
)

message("[Evaluate] 模型比較表、測試集明細與 3 張圖表已全部產出至 outputs/。")
