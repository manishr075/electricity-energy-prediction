# Viva Questions and Answers

## 1. What is the aim of this project?
To predict appliance energy consumption for the next 10-minute interval using a transparent linear regression model, then demonstrate calculus-based training and illustrative cost scenarios.

## 2. Why did you choose linear regression?
It is interpretable, computationally light, and its loss gradients can be derived and implemented manually. It is suitable for demonstrating the connection between calculus and machine learning.

## 3. What is the target variable?
`Appliances` shifted one valid 10-minute observation forward. Its unit in the source dataset is watt-hours (Wh).

## 4. What is a feature?
An input variable used by a model to make a prediction, such as current appliance energy, lighting energy, temperature, humidity, or encoded time.

## 5. What is the hypothesis function?
For multiple features, y_hat = xᵀw + b. The vector w contains feature weights and b is the bias.

## 6. What is a loss function?
A scalar measure of prediction error. This project uses mean squared error: J = (1/n) Σ(y_hatᵢ − yᵢ)².

## 7. Why use squared error?
Squaring prevents positive and negative residuals from cancelling, penalizes larger errors more heavily, and gives a differentiable objective.

## 8. What is a derivative?
It measures how a function changes as an input changes. In optimization, it indicates the local slope of the loss with respect to a parameter.

## 9. What is a partial derivative?
It measures how the loss changes with one parameter while treating other parameters as fixed.

## 10. State the gradients.
For residual vector e = Z w + b − y: ∂J/∂w = (2/n)Zᵀe and ∂J/∂b = 2 mean(e).

## 11. What is gradient descent?
An iterative optimization method that updates parameters in the direction opposite to the gradient to reduce the loss.

## 12. What does the learning rate do?
It sets the step size for parameter updates. Too large a value can cause oscillation or divergence; too small a value makes learning slow.

## 13. Why scale the features?
Features have different units and ranges. Standardization often improves numerical conditioning and makes gradient-descent updates more balanced.

## 14. How do you prevent data leakage?
Use chronological splitting, fit scaling statistics only on training data, and do not use future target information as a feature.

## 15. Why not randomly shuffle the time series?
Random splitting can place later observations in training and earlier observations in testing, making evaluation less representative of future prediction.

## 16. Why check 10-minute intervals?
A next-row target is only a next-10-minute target when consecutive timestamps are exactly ten minutes apart.

## 17. Explain MAE.
Mean absolute error is the average absolute difference between predictions and actual values. It is measured in Wh for this target; lower is generally better.

## 18. Explain RMSE.
Root mean squared error is the square root of mean squared error, measured in Wh. It penalizes large errors more than MAE.

## 19. What does R² mean?
It compares residual variation against a mean-based reference. A value closer to 1 can indicate better fit, while a negative test R² means the model performed worse than the test-set mean reference under the metric definition.

## 20. Why include a baseline?
A complex or carefully coded model is not useful if it cannot beat a simple reference predictor. The training-mean baseline gives a basic benchmark.

## 21. Why compare with scikit-learn?
It provides an independent implementation for sanity-checking the manual regression approach. Differences can reveal implementation, scaling, or optimization issues.

## 22. What is the dataset limitation?
It represents one building in Belgium over a limited period. It cannot by itself prove generalization to Indian households or other buildings.

## 23. How is electricity cost estimated?
Predicted Wh is divided by 1,000 to obtain kWh, then multiplied by a user-entered assumed tariff in ₹/kWh. Actual bills can include slabs, fixed charges, taxes and other rules.

## 24. Does a 20% what-if reduction prove savings will occur?
No. It is a hypothetical arithmetic scenario, not a causal estimate or guarantee that user behavior will reduce consumption by that amount.

## 25. What does convergence mean here?
The loss stabilizes or decreases toward a minimum as updates continue. Check the training-loss plot and verify that values remain finite and do not oscillate or diverge.

## 26. What would you improve next?
Use rolling-origin validation, test additional models, investigate prediction intervals, collect more diverse building data, and add verified local tariff rules.

## 27. Why is calculus essential rather than decorative?
The model's weights and bias are learned using partial derivatives of MSE. Without the gradients and parameter updates, the manual training algorithm would not be implemented.

## 28. What is the difference between prediction and causation?
Prediction estimates an outcome from patterns in data. Causation claims that changing an input produces a change in the outcome; this observational dataset and what-if arithmetic do not establish causation.

## 29. What must you do before reporting performance?
Run the code, inspect the loss curve, read actual values from `outputs/metrics.json`, and document the split and hyperparameters. Do not invent scores.

## 30. What is the project's main contribution?
A reproducible educational pipeline that connects real energy data, calculus-based gradient descent, regression evaluation, and an interactive local dashboard.
