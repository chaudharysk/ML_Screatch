# ML_Scratch
# Linear Regression Derivation (2D)

## Objective
The goal of linear regression is to find the best-fitting straight line through a set of points. The line can be described by the equation:
\[ y = \beta_0 + \beta_1 x \]
where \( \beta_0 \) is the y-intercept and \( \beta_1 \) is the slope of the line.

## Derivation

1. **Formulate the Cost Function**:
   The cost function (also called the loss function) measures how well a given line fits the data points. The most common cost function used is the Mean Squared Error (MSE):
   \[ J(\beta_0, \beta_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]
   where \( \hat{y}_i = \beta_0 + \beta_1 x_i \) is the predicted value.

2. **Rewrite the Cost Function**:
   Expand the squared term in the cost function:
   \[ J(\beta_0, \beta_1) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2 \]

3. **Minimize the Cost Function**:
   To find the best-fitting line, we need to minimize the cost function \( J(\beta_0, \beta_1) \). We do this by taking partial derivatives of \( J \) with respect to \( \beta_0 \) and \( \beta_1 \), and setting them to zero.

   **Partial Derivative with respect to \( \beta_0 \)**:
   \[ \frac{\partial J}{\partial \beta_0} = \frac{1}{n} \sum_{i=1}^n -2(y_i - \beta_0 - \beta_1 x_i) \]
   Set it to zero:
   \[ \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i) = 0 \]
   \[ n\beta_0 + \beta_1 \sum_{i=1}^n x_i = \sum_{i=1}^n y_i \]
   \[ \beta_0 + \beta_1 \bar{x} = \bar{y} \]
   where \( \bar{x} \) and \( \bar{y} \) are the means of \( x \) and \( y \), respectively.

   **Partial Derivative with respect to \( \beta_1 \)**:
   \[ \frac{\partial J}{\partial \beta_1} = \frac{1}{n} \sum_{i=1}^n -2x_i(y_i - \beta_0 - \beta_1 x_i) \]
   Set it to zero:
   \[ \sum_{i=1}^n x_i (y_i - \beta_0 - \beta_1 x_i) = 0 \]
   \[ \beta_0 \sum_{i=1}^n x_i + \beta_1 \sum_{i=1}^n x_i^2 = \sum_{i=1}^n x_i y_i \]

4. **Solve the System of Equations**:
   Substitute \( \beta_0 = \bar{y} - \beta_1 \bar{x} \) into the second equation:
   \[ (\bar{y} - \beta_1 \bar{x}) \sum_{i=1}^n x_i + \beta_1 \sum_{i=1}^n x_i^2 = \sum_{i=1}^n x_i y_i \]
   Simplify:
   \[ \bar{y} \sum_{i=1}^n x_i - \beta_1 \bar{x} \sum_{i=1}^n x_i + \beta_1 \sum_{i=1}^n x_i^2 = \sum_{i=1}^n x_i y_i \]
   \[ \beta_1 \left(\sum_{i=1}^n x_i^2 - \frac{(\sum_{i=1}^n x_i)^2}{n} \right) = \sum_{i=1}^n x_i y_i - \frac{\sum_{i=1}^n x_i \sum_{i=1}^n y_i}{n} \]
   \[ \beta_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} \]

5. **Determine \( \beta_0 \)**:
   Once \( \beta_1 \) is known, substitute it back into:
   \[ \beta_0 = \bar{y} - \beta_1 \bar{x} \]

## Summary
The final coefficients for the best-fitting line are:
\[ \beta_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} \]
\[ \beta_0 = \bar{y} - \beta_1 \bar{x} \]

# Multilinear Regression Derivation

## Model
The equation for a multilinear regression model with \( p \) independent variables is:
\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon \]
where:
- \( y \) is the dependent variable.
- \( \beta_0 \) is the intercept.
- \( \beta_1, \beta_2, \ldots, \beta_p \) are the coefficients for the independent variables \( x_1, x_2, \ldots, x_p \).
- \( \epsilon \) is the error term.

## Objective
The goal is to estimate the coefficients \( \beta_0, \beta_1, \ldots, \beta_p \) such that the sum of the squared differences between the observed and predicted values of \( y \) is minimized.

## Derivation

1. **Formulate the Cost Function**:
   The cost function (often the Mean Squared Error) measures how well the model fits the data. It is given by:
   \[ J(\beta_0, \beta_1, \ldots, \beta_p) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]
   where \( \hat{y}_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} \) is the predicted value.

2. **Rewrite the Cost Function in Matrix Form**:
   To simplify the calculations, we rewrite the cost function using matrix notation. Let:
   \[ \mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}, \quad \mathbf{X} = \begin{pmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix}, \quad \boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_p \end{pmatrix} \]
   The predicted values can be written as:
   \[ \hat{\mathbf{y}} = \mathbf{X} \boldsymbol{\beta} \]
   The cost function becomes:
   \[ J(\boldsymbol{\beta}) = \frac{1}{n} (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta}) \]

3. **Minimize the Cost Function**:
   To find the optimal \( \boldsymbol{\beta} \), we take the derivative of \( J(\boldsymbol{\beta}) \) with respect to \( \boldsymbol{\beta} \) and set it to zero:
   \[ \frac{\partial J}{\partial \boldsymbol{\beta}} = \frac{2}{n} \mathbf{X}^T (\mathbf{X} \boldsymbol{\beta} - \mathbf{y}) = 0 \]
   Simplifying, we get:
   \[ \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y} \]

4. **Solve for \( \boldsymbol{\beta} \)**:
   Provided \( \
