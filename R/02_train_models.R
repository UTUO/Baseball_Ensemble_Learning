# ==============================================================================
# 02_train_models.R
# 僅使用訓練集訓練 Linear Regression、CART、Random Forest、XGBoost 四個模型
# ==============================================================================

# 1. Multiple Linear Regression（傳統統計基準模型，比照課程範例使用 glm）
lm_fit <- glm(
  Logsalary ~ .,
  family = gaussian(link = "identity"),
  data = train_data
)

# 2. CART Regression Tree（比照 CART_Baseball.r 與 CART_Test.r，修正為僅用 train_data）
set.seed(123)
cart_fit <- rpart::rpart(
  Logsalary ~ .,
  method = "anova",
  data = train_data
)

# 3. Random Forest Regression（比照 RF_Test.r，mtry = floor(19 / 3) = 6）
set.seed(123)
rf_fit <- randomForest::randomForest(
  Logsalary ~ .,
  data = train_data,
  ntree = 500,
  mtry = 6,
  nodesize = 5,
  importance = TRUE
)

# 4. XGBoost Regression（比照 XGBoost_Test.r 迴歸參數，絕不於訓練階段傳入測試集）
dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgboost::xgb.DMatrix(data = x_test, label = y_test)

xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 3,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  tree_method = "hist",
  nthread = 1,
  seed = 123
)

set.seed(123)
xgb_fit <- xgboost::xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 200,
  verbose = 0
)

message("[Train] 四個模型（Linear Regression, CART, Random Forest, XGBoost）訓練完成。")
